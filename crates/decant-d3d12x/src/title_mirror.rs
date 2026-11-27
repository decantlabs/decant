//! The tables behind `title_mem`, as plain safe code with no D3D12 in it: the reservations and
//! views of `xmem.dll` as its journal reports them, the slots reservations get in the window
//! buffers, the tile ranges that make the window show the pool pages, and the arithmetic of a
//! translated address.
//!
//! A pool page (the engine's page index) gets a mirror tile when it first has bytes for the GPU (`give_tile`:
//! its first copy, or a buffer placed over it), in that order, and keeps it: mirror tile `t` is tile
//! `t % HEAP_PAGES` of pool heap `t / HEAP_PAGES`. So the heaps grow with the pages the title wrote, not
//! with the pool nor with what it maps: a page nobody touched (a render target, the unused part of a
//! streaming pool) costs no tile, its window tiles show nothing and read as zeros, as the page does, and
//! they are mapped to its tile once it gets one. A reservation gets a slot of its own size in a window buffer when its first page is mapped. Title
//! byte `va` of a reservation at `base` is at `slot offset + (va - base)` in the window buffer, so a
//! reservation is contiguous on the GPU over gaps, aliases and scattered pages.

use std::collections::{BTreeMap, HashMap};

/// The page of the title, the tile of a reserved buffer and the tile of a heap: 64 KiB.
pub const PAGE: u64 = 0x1_0000;
/// Pool pages in one pool heap: 16 MiB.
pub const HEAP_PAGES: u32 = 256;
/// The largest reservation a window buffer can hold, tiled resources tier 4: 40 bits of GPU address.
pub const MAX_SLOT_BYTES: u64 = 1 << 40;

/// The pool heap and the tile in it of mirror tile `tile`.
pub fn heap_of(tile: u32) -> (u32, u32) {
    (tile / HEAP_PAGES, tile % HEAP_PAGES)
}

/// No mirror tile yet.
const NO_TILE: u32 = u32::MAX;

/// The pages that hold the `len` bytes at `va`: the address of the first one and how many.
pub fn page_span(va: u64, len: u64) -> (u64, u32) {
    let first = va & !(PAGE - 1);
    let end = va.saturating_add(len.max(1));
    (first, (end - first).div_ceil(PAGE).min(u64::from(u32::MAX)) as u32)
}

/// The page-aligned base, lead offset (`0..PAGE`), and page count covering `[va, va + bytes)`.
/// Returns `None` for a zero-byte span or when the end address or page count overflows.
pub fn checked_page_span(va: u64, bytes: u64) -> Option<(u64, u64, u32)> {
    if bytes == 0 {
        return None;
    }
    let aligned_va = va & !(PAGE - 1);
    let base_offset = va - aligned_va;
    let end = va.checked_add(bytes)?;
    let pages = u32::try_from((end - aligned_va).div_ceil(PAGE)).ok()?;
    (pages != 0).then_some((aligned_va, base_offset, pages))
}

/// One change of the address space, as the journal of `xmem.dll` says it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    /// `on_demand`: a commit of the title whose pages are mapped as the GPU is given addresses in them
    /// (`Mirror::wants_use`, then `DecantXmemGpuUse` in `xmem.dll`).
    Reserve { va: u64, pages: u32, on_demand: bool },
    Release { va: u64 },
    Map { va: u64, pages: u32, first_page: u32 },
    Unmap { va: u64, pages: u32 },
}

/// What a window tile shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Mirror tile.
    Heap(u32),
    /// Nothing: reads give 0 and writes are dropped.
    Null,
}

/// One `UpdateTileMappings` call: the tile ranges of window buffer `buffer` that go to one pool heap
/// (`heap` is `Some`, `heap_tiles[i]` is where range `i` starts in it) or to nothing (`None`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileCall {
    pub buffer: usize,
    pub heap: Option<u32>,
    /// `(first window tile, tile count)` of each range.
    pub ranges: Vec<(u32, u32)>,
    pub heap_tiles: Vec<u32>,
}

/// Where a title span is in the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    pub buffer: usize,
    /// The byte offset of the span in the window buffer.
    pub offset: u64,
    /// The bytes from the end of the span to the end of its slot.
    pub left: u64,
}

/// Mirror tiles covering a title byte span `[va, va + bytes)` inside one reservation, together with
/// the page-aligned start address and byte offset of `va` inside tile 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacedTiles {
    /// Page-aligned title address of tile 0 (`va & !(PAGE - 1)`).
    pub aligned_va: u64,
    /// Byte offset of `va` inside tile 0 (`0..PAGE`).
    pub base_offset: u64,
    /// Mirror tile index for each 64 KiB page covering `[va, va + bytes)`.
    pub tiles: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    buffer: usize,
    offset: u64,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct View {
    pages: u32,
    first_page: u32,
}

#[derive(Debug)]
struct Reservation {
    bytes: u64,
    slot: Option<Slot>,
    /// The reservation is larger than any buffer can be: logged once, never retried.
    refused: bool,
    /// Its pages come as the GPU is given addresses in them.
    on_demand: bool,
    /// Mapped runs by title address.
    views: BTreeMap<u64, View>,
}

#[derive(Debug)]
struct Buffer {
    bytes: u64,
    /// It holds the slots of commits whose pages come on demand and no other: those commits hold the data the
    /// GPU is given by address, and packed together they fit the 4 GiB a translated shader reaches.
    on_demand: bool,
    /// Free ranges by offset.
    free: BTreeMap<u64, u64>,
}

impl Buffer {
    /// Takes `bytes` from the first free range that holds them.
    fn take(&mut self, bytes: u64) -> Option<u64> {
        let (&offset, &len) = self.free.iter().find(|&(_, &len)| len >= bytes)?;
        self.free.remove(&offset);
        if len > bytes {
            self.free.insert(offset + bytes, len - bytes);
        }
        Some(offset)
    }

    /// Gives a range back and joins it with the free ranges next to it.
    fn give(&mut self, offset: u64, bytes: u64) {
        let (mut start, mut len) = (offset, bytes);
        if let Some((&before, &blen)) = self.free.range(..offset).next_back()
            && before + blen == offset
        {
            self.free.remove(&before);
            start = before;
            len += blen;
        }
        if let Some(&after) = self.free.get(&(offset + bytes)) {
            self.free.remove(&(offset + bytes));
            len += after;
        }
        self.free.insert(start, len);
    }
}

/// Maximum pool-page index kept in the flat `owners_dense` vector; larger sparse indices spill
/// into `owners_sparse`.
const DENSE_OWNER_LIMIT: usize = 4_000_000;

/// Owners of one pool page across window tiles: inline for the common 1-owner case, boxed `Vec`
/// only when a physical pool page is aliased into multiple virtual tiles.
#[derive(Clone, Debug, Default)]
enum PageOwners {
    #[default]
    None,
    One(u32, u32),
    Many(Vec<(usize, u32)>),
}

impl PageOwners {
    fn add(&mut self, at: (usize, u32), buf_u32: Option<u32>) {
        match self {
            Self::None => {
                if let Some(b) = buf_u32 {
                    *self = Self::One(b, at.1);
                } else {
                    *self = Self::Many(vec![at]);
                }
            }
            Self::One(b, t) => {
                let first = (*b as usize, *t);
                if first != at {
                    *self = Self::Many(vec![first, at]);
                }
            }
            Self::Many(v) => {
                if !v.contains(&at) {
                    v.push(at);
                }
            }
        }
    }

    fn remove(&mut self, at: (usize, u32)) -> bool {
        match self {
            Self::None => true,
            Self::One(b, t) => {
                if (*b as usize, *t) == at {
                    *self = Self::None;
                    true
                } else {
                    false
                }
            }
            Self::Many(v) => {
                v.retain(|&o| o != at);
                if v.is_empty() {
                    *self = Self::None;
                    true
                } else if v.len() == 1
                    && let Ok(b) = u32::try_from(v[0].0)
                {
                    *self = Self::One(b, v[0].1);
                    false
                } else {
                    false
                }
            }
        }
    }

    fn for_each(&self, mut f: impl FnMut((usize, u32))) {
        match self {
            Self::None => {}
            Self::One(b, t) => f((*b as usize, *t)),
            Self::Many(v) => {
                for &at in v.iter() {
                    f(at);
                }
            }
        }
    }
}

/// Contiguous run of pending window tile updates within one pool heap (or all `Null`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingRun {
    pages: u32,
    target: Target,
}

fn advance_target(t: Target, delta: u32) -> Target {
    match t {
        Target::Null => Target::Null,
        Target::Heap(p) => Target::Heap(p + delta),
    }
}

fn can_merge_runs(left: Target, left_pages: u32, right: Target) -> bool {
    match (left, right) {
        (Target::Null, Target::Null) => true,
        (Target::Heap(lp), Target::Heap(rp)) => lp + left_pages == rp && lp / HEAP_PAGES == rp / HEAP_PAGES,
        _ => false,
    }
}

fn pending_insert_chunk(pending: &mut BTreeMap<(usize, u32), PendingRun>, buffer: usize, start: u32, count: u32, target: Target) {
    if count == 0 {
        return;
    }
    let end = start + count;
    // Split any run starting strictly before `start` that overlaps `[start, end)`.
    if let Some((&(b, lk), &lrun)) = pending.range(..(buffer, start)).next_back()
        && b == buffer
        && lk + lrun.pages > start
    {
        pending.remove(&(b, lk));
        pending.insert((b, lk), PendingRun { pages: start - lk, target: lrun.target });
        if lk + lrun.pages > end {
            pending.insert((b, end), PendingRun { pages: lk + lrun.pages - end, target: advance_target(lrun.target, end - lk) });
        }
    }
    // Remove/trim any runs starting in `[start, end)`.
    let overlapping: Vec<(u32, PendingRun)> = pending.range((buffer, start)..(buffer, end)).map(|(&(_, k), &v)| (k, v)).collect();
    for (k, run) in overlapping {
        pending.remove(&(buffer, k));
        if k + run.pages > end {
            pending.insert((buffer, end), PendingRun { pages: k + run.pages - end, target: advance_target(run.target, end - k) });
        }
    }
    // Merge with left neighbor if adjacent and compatible within the same heap.
    let (mut ins_start, mut ins_pages, mut ins_target) = (start, count, target);
    if let Some((&(b, lk), &lrun)) = pending.range(..(buffer, start)).next_back()
        && b == buffer
        && lk + lrun.pages == start
        && can_merge_runs(lrun.target, lrun.pages, ins_target)
    {
        pending.remove(&(b, lk));
        ins_start = lk;
        ins_pages += lrun.pages;
        ins_target = lrun.target;
    }
    // Merge with right neighbor if adjacent and compatible within the same heap.
    if let Some(&rrun) = pending.get(&(buffer, end))
        && can_merge_runs(ins_target, ins_pages, rrun.target)
    {
        pending.remove(&(buffer, end));
        ins_pages += rrun.pages;
    }
    pending.insert((buffer, ins_start), PendingRun { pages: ins_pages, target: ins_target });
}

fn pending_set(pending: &mut BTreeMap<(usize, u32), PendingRun>, buffer: usize, start: u32, count: u32, target: Target) {
    let mut done = 0u32;
    while done < count {
        let sub_target = advance_target(target, done);
        let sub_count = match sub_target {
            Target::Null => count - done,
            Target::Heap(p) => (HEAP_PAGES - (p % HEAP_PAGES)).min(count - done),
        };
        pending_insert_chunk(pending, buffer, start + done, sub_count, sub_target);
        done += sub_count;
    }
}

#[derive(Debug)]
pub struct Mirror {
    /// The size of a window buffer.
    window_bytes: u64,
    reservations: BTreeMap<u64, Reservation>,
    buffers: Vec<Buffer>,
    /// Non-overlapping pending tile update runs keyed by `(window buffer, first window tile)`.
    pending: BTreeMap<(usize, u32), PendingRun>,
    generation: u32,
    /// The mirror tile of each pool page, `NO_TILE` before it has bytes (`give_tile`).
    tiles: Vec<u32>,
    /// The pool page of each mirror tile, in tile order.
    tile_pages: Vec<u32>,
    /// Dense owners table indexed by pool page (`< DENSE_OWNER_LIMIT`).
    owners_dense: Vec<PageOwners>,
    /// Overflow owners map for pool pages `>= DENSE_OWNER_LIMIT`.
    owners_sparse: HashMap<u32, PageOwners>,
    /// The mirror tiles given so far.
    next_tile: u32,
}

impl Mirror {
    pub fn new(window_bytes: u64) -> Mirror {
        Mirror {
            window_bytes: window_bytes.max(PAGE),
            reservations: BTreeMap::new(),
            buffers: Vec::new(),
            pending: BTreeMap::new(),
            generation: 0,
            tiles: Vec::new(),
            tile_pages: Vec::new(),
            owners_dense: Vec::new(),
            owners_sparse: HashMap::new(),
            next_tile: 0,
        }
    }

    /// The mirror tile of pool page `page`, given now when it has none: the page has bytes for the GPU (it
    /// is about to be copied, or a buffer is placed over it). Every window tile that shows the page is
    /// mapped to the new tile at the next update.
    pub fn give_tile(&mut self, page: u32) -> u32 {
        let i = page as usize;
        if self.tiles.len() <= i {
            self.tiles.resize(i + 1, NO_TILE);
        }
        if self.tiles[i] == NO_TILE {
            let tile = self.next_tile;
            self.tiles[i] = tile;
            self.tile_pages.push(page);
            self.next_tile += 1;
            crate::mem_owners::add(
                crate::mem_owners::Kind::MirrorTile,
                crate::mem_owners::Segment::System,
                1,
                PAGE,
            );
            let target = Target::Heap(tile);
            if i < self.owners_dense.len() {
                let pending = &mut self.pending;
                self.owners_dense[i].for_each(|(b, t)| pending_set(pending, b, t, 1, target));
            } else if i >= DENSE_OWNER_LIMIT
                && let Some(owners) = self.owners_sparse.get(&page)
            {
                let pending = &mut self.pending;
                owners.for_each(|(b, t)| pending_set(pending, b, t, 1, target));
            }
        }
        self.tiles[i]
    }

    /// Whether every mapped page of `[va, va + len)` (clipped to its reservation) already has a mirror tile.
    pub fn has_tiles_for_span(&self, va: u64, len: u64) -> bool {
        let Some((base, r)) = self.holding(va) else { return false };
        let end = va.saturating_add(len.max(1)).min(base + r.bytes);
        let mut page = va & !(PAGE - 1);
        while page < end {
            if let Some((&v, view)) = r.views.range(..=page).next_back()
                && page < v + u64::from(view.pages) * PAGE
            {
                let pool_page = view.first_page + ((page - v) / PAGE) as u32;
                if self.tile_of(pool_page).is_none() {
                    return false;
                }
            }
            page += PAGE;
        }
        true
    }

    /// Gives a mirror tile to every mapped page of `[va, va + len)` (clipped to its reservation) that has none.
    pub fn give_span(&mut self, va: u64, len: u64) {
        let Some((base, r)) = self.holding(va) else { return };
        let end = va.saturating_add(len.max(1)).min(base + r.bytes);
        let mut pages = Vec::new();
        let mut page = va & !(PAGE - 1);
        while page < end {
            if let Some((&v, view)) = r.views.range(..=page).next_back()
                && page < v + u64::from(view.pages) * PAGE
            {
                pages.push(view.first_page + ((page - v) / PAGE) as u32);
            }
            page += PAGE;
        }
        for page in pages {
            self.give_tile(page);
        }
    }

    fn ensure_owner_capacity(&mut self, first_page: u32, pages: u32) {
        let dense_end = first_page.saturating_add(pages).min(DENSE_OWNER_LIMIT as u32) as usize;
        if self.owners_dense.len() < dense_end {
            self.owners_dense.resize_with(dense_end, PageOwners::default);
        }
    }

    /// Records window tiles `first_tile..first_tile + pages` as showing pool pages
    /// `first_page..first_page + pages` and emits range-coalesced pending tile updates.
    fn show_span(&mut self, buffer: usize, first_tile: u32, first_page: u32, pages: u32) {
        if pages == 0 {
            return;
        }
        self.ensure_owner_capacity(first_page, pages);
        let buf_u32 = u32::try_from(buffer).ok();
        let tiles_len = self.tiles.len();
        let mut run_start = 0u32;
        let mut run_target: Option<Target> = None;
        for k in 0..pages {
            let page = first_page + k;
            let at = (buffer, first_tile + k);
            let p_idx = page as usize;
            if p_idx < DENSE_OWNER_LIMIT {
                self.owners_dense[p_idx].add(at, buf_u32);
            } else {
                self.owners_sparse.entry(page).or_default().add(at, buf_u32);
            }
            let target = if p_idx < tiles_len && self.tiles[p_idx] != NO_TILE {
                Target::Heap(self.tiles[p_idx])
            } else {
                Target::Null
            };
            match run_target {
                None => {
                    run_start = k;
                    run_target = Some(target);
                }
                Some(rt) if can_merge_runs(rt, k - run_start, target) => {}
                Some(rt) => {
                    pending_set(&mut self.pending, buffer, first_tile + run_start, k - run_start, rt);
                    run_start = k;
                    run_target = Some(target);
                }
            }
        }
        if let Some(rt) = run_target {
            pending_set(&mut self.pending, buffer, first_tile + run_start, pages - run_start, rt);
        }
    }

    /// Window tiles from `first_tile` no longer show the `pages` pool pages from `first_page`: they show
    /// nothing.
    fn hide(&mut self, buffer: usize, first_tile: u32, pages: u32, first_page: u32) {
        if pages == 0 {
            return;
        }
        let dense_len = self.owners_dense.len();
        for k in 0..pages {
            let page = first_page + k;
            let at = (buffer, first_tile + k);
            let p_idx = page as usize;
            if p_idx < dense_len {
                self.owners_dense[p_idx].remove(at);
            } else if p_idx >= DENSE_OWNER_LIMIT
                && let Some(owners) = self.owners_sparse.get_mut(&page)
                && owners.remove(at)
            {
                self.owners_sparse.remove(&page);
            }
        }
        pending_set(&mut self.pending, buffer, first_tile, pages, Target::Null);
    }

    /// The mirror tiles given so far.
    pub fn tiles_given(&self) -> u32 {
        self.next_tile
    }

    /// The mirror tile of pool page `page`, `None` when it was never mapped.
    pub fn tile_of(&self, page: u32) -> Option<u32> {
        self.tiles.get(page as usize).copied().filter(|&t| t != NO_TILE)
    }

    /// The pool page of mirror tile `tile`, `None` past the tiles given so far.
    pub fn page_of_tile(&self, tile: u32) -> Option<u32> {
        self.tile_pages.get(tile as usize).copied()
    }

    /// Pool pages whose mirror tile falls in pool heap `heap`, in mirror tile order.
    pub fn pages_in_heap(&self, heap: u32) -> Vec<u32> {
        let Some(first_tile) = heap.checked_mul(HEAP_PAGES) else {
            return Vec::new();
        };
        let start = (first_tile as usize).min(self.tile_pages.len());
        let end = (first_tile.saturating_add(HEAP_PAGES) as usize).min(self.tile_pages.len());
        self.tile_pages[start..end].to_vec()
    }

    /// Re-queues pending tile updates for every live window tile showing a pool page of `heap`.
    /// Used when a transiently deferred pool heap is materialized later. Returns how many window
    /// tile mappings were queued.
    pub fn remap_heap_tiles(&mut self, heap: u32) -> usize {
        let Some(first_tile) = heap.checked_mul(HEAP_PAGES) else {
            return 0;
        };
        let start = (first_tile as usize).min(self.tile_pages.len());
        let end = (first_tile.saturating_add(HEAP_PAGES) as usize).min(self.tile_pages.len());
        let mut queued = 0usize;
        for tile_idx in start..end {
            let page = self.tile_pages[tile_idx];
            let target = Target::Heap(tile_idx as u32);
            let p_idx = page as usize;
            if p_idx < self.owners_dense.len() {
                let pending = &mut self.pending;
                self.owners_dense[p_idx].for_each(|(b, t)| {
                    pending_set(pending, b, t, 1, target);
                    queued += 1;
                });
            } else if p_idx >= DENSE_OWNER_LIMIT
                && let Some(owners) = self.owners_sparse.get(&page)
            {
                let pending = &mut self.pending;
                owners.for_each(|(b, t)| {
                    pending_set(pending, b, t, 1, target);
                    queued += 1;
                });
            }
        }
        queued
    }

    /// Changes whenever a slot is given or freed. A mapping inside a slot does not change any address.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The size of each window buffer made so far, by index.
    pub fn buffer_bytes(&self) -> Vec<u64> {
        self.buffers.iter().map(|b| b.bytes).collect()
    }

    /// One slot per entry: window buffer, offset in it and bytes.
    #[cfg(test)]
    pub fn extents(&self) -> Vec<(usize, u64, u64)> {
        self.slots().into_iter().map(|(_, b, o, n)| (b, o, n)).collect()
    }

    /// One slot per entry with the title address of its reservation: base, window buffer, offset, bytes.
    pub fn slots(&self) -> Vec<(u64, usize, u64, u64)> {
        self.reservations.iter().filter_map(|(&base, r)| r.slot.map(|s| (base, s.buffer, s.offset, s.bytes))).collect()
    }

    fn alloc(&mut self, bytes: u64, on_demand: bool) -> Option<Slot> {
        for (buffer, b) in self.buffers.iter_mut().enumerate().filter(|(_, b)| b.on_demand == on_demand) {
            if let Some(offset) = b.take(bytes) {
                return Some(Slot { buffer, offset, bytes });
            }
        }
        if bytes > MAX_SLOT_BYTES {
            return None;
        }
        // A new buffer: the window size, or the reservation's own size when that is larger.
        let size = bytes.max(self.window_bytes);
        self.buffers.push(Buffer { bytes: size, on_demand, free: BTreeMap::from([(0, size)]) });
        let buffer = self.buffers.len() - 1;
        let offset = self.buffers[buffer].take(bytes)?;
        Some(Slot { buffer, offset, bytes })
    }

    /// The reservation that holds `va`, with its base.
    fn holding(&self, va: u64) -> Option<(u64, &Reservation)> {
        let (&base, r) = self.reservations.range(..=va).next_back()?;
        (va - base < r.bytes).then_some((base, r))
    }

    /// The window tile of title address `va` of the reservation at `base` in `slot`.
    fn window_tile(slot: Slot, base: u64, va: u64) -> u32 {
        ((slot.offset + (va - base)) / PAGE) as u32
    }

    /// Takes the mapped runs out of `[start, end)`, cutting the ones that stick out. Returns what was
    /// removed as `(va, pages, first pool page)`.
    fn cut(views: &mut BTreeMap<u64, View>, start: u64, end: u64) -> Vec<(u64, u32, u32)> {
        let mut touched: Vec<(u64, View)> = Vec::new();
        if let Some((&v, &view)) = views.range(..start).next_back()
            && v + u64::from(view.pages) * PAGE > start
        {
            touched.push((v, view));
        }
        for (&v, &view) in views.range(start..end) {
            touched.push((v, view));
        }
        let mut removed = Vec::with_capacity(touched.len());
        for (v, view) in touched {
            let vend = v + u64::from(view.pages) * PAGE;
            views.remove(&v);
            if v < start {
                views.insert(v, View { pages: ((start - v) / PAGE) as u32, first_page: view.first_page });
            }
            if vend > end {
                views.insert(end, View { pages: ((vend - end) / PAGE) as u32, first_page: view.first_page + ((end - v) / PAGE) as u32 });
            }
            let (from, to) = (v.max(start), vend.min(end));
            removed.push((from, ((to - from) / PAGE) as u32, view.first_page + ((from - v) / PAGE) as u32));
        }
        removed
    }

    /// Applies one change. Returns the size of a reservation that was refused (larger than a buffer
    /// can be), so that the caller logs it once.
    pub fn apply(&mut self, change: Change) -> Option<u64> {
        match change {
            Change::Reserve { va, pages, on_demand } => {
                self.release(va);
                self.reservations.insert(va, Reservation { bytes: u64::from(pages) * PAGE, slot: None, refused: false, on_demand, views: BTreeMap::new() });
                None
            }
            Change::Release { va } => {
                self.release(va);
                None
            }
            Change::Map { va, pages, first_page } => {
                let (base, bytes) = self.holding(va).map(|(b, r)| (b, r.bytes))?;
                let pages = pages.min(((base + bytes - va) / PAGE) as u32);
                let mut refused = None;
                let needs_slot = self.reservations.get(&base).is_some_and(|r| r.slot.is_none() && !r.refused);
                if needs_slot {
                    let on_demand = self.reservations.get(&base).is_some_and(|r| r.on_demand);
                    match self.alloc(bytes, on_demand) {
                        Some(slot) => {
                            self.generation += 1;
                            self.reservations.get_mut(&base)?.slot = Some(slot);
                        }
                        None => {
                            self.reservations.get_mut(&base)?.refused = true;
                            refused = Some(bytes);
                        }
                    }
                }
                let r = self.reservations.get_mut(&base)?;
                let Some(slot) = r.slot else { return refused };
                let removed = Self::cut(&mut r.views, va, va + u64::from(pages) * PAGE);
                r.views.insert(va, View { pages, first_page });
                for (at, n, page) in removed {
                    self.hide(slot.buffer, Self::window_tile(slot, base, at), n, page);
                }
                let first_tile = Self::window_tile(slot, base, va);
                self.show_span(slot.buffer, first_tile, first_page, pages);
                refused
            }
            Change::Unmap { va, pages } => {
                let base = self.holding(va).map(|(b, _)| b)?;
                let r = self.reservations.get_mut(&base)?;
                let slot = r.slot?;
                let end = (va + u64::from(pages) * PAGE).min(base + r.bytes);
                for (at, n, page) in Self::cut(&mut r.views, va, end) {
                    self.hide(slot.buffer, Self::window_tile(slot, base, at), n, page);
                }
                None
            }
        }
    }

    /// Forgets the reservation at `va`: its mapped tiles show nothing and its slot is free again.
    fn release(&mut self, va: u64) {
        let Some(r) = self.reservations.remove(&va) else { return };
        let Some(slot) = r.slot else { return };
        for (&at, view) in &r.views {
            self.hide(slot.buffer, Self::window_tile(slot, va, at), view.pages, view.first_page);
        }
        self.buffers[slot.buffer].give(slot.offset, slot.bytes);
        self.generation += 1;
    }

    /// Forgets every reservation, before the tables are built again from a snapshot.
    pub fn reset(&mut self) {
        let bases: Vec<u64> = self.reservations.keys().copied().collect();
        for base in bases {
            self.release(base);
        }
    }

    /// Where `[va, va + len)` is in the window: the reservation has a slot, the span lies inside it and the
    /// page of `va` is mapped.
    pub fn translate(&self, va: u64, len: u64) -> Option<Placed> {
        let (base, r) = self.holding(va)?;
        let slot = r.slot?;
        let at = va - base;
        let end = at.checked_add(len.max(1)).filter(|&e| e <= r.bytes)?;
        let page = va & !(PAGE - 1);
        let (&v, view) = r.views.range(..=page).next_back()?;
        (page < v + u64::from(view.pages) * PAGE).then_some(Placed { buffer: slot.buffer, offset: slot.offset + at, left: slot.bytes - end })
    }

    /// Whether `[va, va + len)` lies in a reservation whose pages come on demand and a page of it (inside
    /// the reservation) is not mapped yet: `xmem.dll` must be asked for those pages first.
    pub fn wants_use(&self, va: u64, len: u64) -> bool {
        let Some((base, r)) = self.holding(va) else { return false };
        if !r.on_demand || r.refused {
            return false;
        }
        let end = va.saturating_add(len.max(1)).min(base + r.bytes);
        let mut page = va & !(PAGE - 1);
        while page < end {
            match r.views.range(..=page).next_back() {
                Some((&v, view)) if page < v + u64::from(view.pages) * PAGE => page = v + u64::from(view.pages) * PAGE,
                _ => return true,
            }
        }
        false
    }

    /// Read-only version of [`mirror_tiles`](Self::mirror_tiles): returns `Some` only when every page
    /// in the span is mapped in one reservation and already has a mirror tile.
    pub fn mirror_tiles_read(&self, va: u64, pages: u32) -> Option<Vec<u32>> {
        if !va.is_multiple_of(PAGE) || pages == 0 {
            return None;
        }
        let (base, r) = self.holding(va)?;
        if va - base + u64::from(pages) * PAGE > r.bytes {
            return None;
        }
        let mut out = Vec::with_capacity(pages as usize);
        for k in 0..u64::from(pages) {
            let page = va + k * PAGE;
            let (&v, view) = r.views.range(..=page).next_back()?;
            let index = (page - v) / PAGE;
            if index >= u64::from(view.pages) {
                return None;
            }
            out.push(self.tile_of(view.first_page + index as u32)?);
        }
        Some(out)
    }

    /// The mirror tile of each of the `pages` title pages from `va` (on a page), when all of them lie in
    /// one reservation and are mapped, given now to the pages that have none. A buffer placed there is a
    /// reserved buffer over these tiles, and what the GPU writes into it must stay.
    pub fn mirror_tiles(&mut self, va: u64, pages: u32) -> Option<Vec<u32>> {
        if !va.is_multiple_of(PAGE) || pages == 0 {
            return None;
        }
        let (base, r) = self.holding(va)?;
        if va - base + u64::from(pages) * PAGE > r.bytes {
            return None;
        }
        let pool_pages: Option<Vec<u32>> = (0..u64::from(pages))
            .map(|k| {
                let page = va + k * PAGE;
                let (&v, view) = r.views.range(..=page).next_back()?;
                let index = (page - v) / PAGE;
                (index < u64::from(view.pages)).then_some(view.first_page + index as u32)
            })
            .collect();
        Some(pool_pages?.into_iter().map(|page| self.give_tile(page)).collect())
    }

    /// Read-only version of [`mirror_tiles_span`](Self::mirror_tiles_span): returns `Some` only
    /// when every 64 KiB page covering `[va, va + bytes)` is mapped in one reservation and already
    /// has a mirror tile.
    pub fn mirror_tiles_span_read(&self, va: u64, bytes: u64) -> Option<PlacedTiles> {
        let (aligned_va, base_offset, pages) = checked_page_span(va, bytes)?;
        let tiles = self.mirror_tiles_read(aligned_va, pages)?;
        Some(PlacedTiles { aligned_va, base_offset, tiles })
    }

    /// Mirror tiles covering `[va, va + bytes)` at arbitrary byte alignment inside one reservation,
    /// giving mirror tiles to mapped pages that have none. When `va` is not page-aligned, the
    /// returned `base_offset` (`0..PAGE`) locates `va` inside tile 0.
    pub fn mirror_tiles_span(&mut self, va: u64, bytes: u64) -> Option<PlacedTiles> {
        let (aligned_va, base_offset, pages) = checked_page_span(va, bytes)?;
        let tiles = self.mirror_tiles(aligned_va, pages)?;
        Some(PlacedTiles { aligned_va, base_offset, tiles })
    }

    /// Whether one of the `pages` title pages from `va` (on a page) shows a mirror tile of `written`:
    /// `None` when they do not all lie mapped in one reservation, as `mirror_tiles` answers. Gives no
    /// tile and allocates nothing: a page without a tile was not copied, so it is not in `written`.
    pub fn touches(&self, va: u64, pages: u32, written: &std::collections::HashSet<u32>) -> Option<bool> {
        if !va.is_multiple_of(PAGE) || pages == 0 {
            return None;
        }
        let (base, r) = self.holding(va)?;
        if va - base + u64::from(pages) * PAGE > r.bytes {
            return None;
        }
        let mut hit = false;
        for k in 0..u64::from(pages) {
            let page = va + k * PAGE;
            let (&v, view) = r.views.range(..=page).next_back()?;
            let index = (page - v) / PAGE;
            if index >= u64::from(view.pages) {
                return None;
            }
            hit = hit || self.tile_of(view.first_page + index as u32).is_some_and(|t| written.contains(&t));
        }
        Some(hit)
    }

    /// Whether a window tile of the `len` bytes at `offset` of window buffer `buffer` has a mapping
    /// change that the next update applies.
    pub fn pending_in(&self, buffer: usize, offset: u64, len: u64) -> bool {
        let first = (offset / PAGE) as u32;
        let last = (offset.saturating_add(len.max(1)) - 1) / PAGE;
        let last = u32::try_from(last).unwrap_or(u32::MAX);
        if self.pending.range((buffer, first)..=(buffer, last)).next().is_some() {
            return true;
        }
        self.pending
            .range(..(buffer, first))
            .next_back()
            .is_some_and(|(&(b, k), run)| b == buffer && k + run.pages > first)
    }

    /// The tile mappings to apply now, one call for each window buffer and pool heap (one more for the
    /// tiles that show nothing), and forgets them.
    pub fn drain_updates(&mut self) -> Vec<TileCall> {
        let mut calls: Vec<TileCall> = Vec::new();
        for ((buffer, tile), PendingRun { pages, target }) in std::mem::take(&mut self.pending) {
            let heap = match target {
                Target::Heap(p) => Some(heap_of(p).0),
                Target::Null => None,
            };
            let call = match calls.iter().position(|c| c.buffer == buffer && c.heap == heap) {
                Some(i) => i,
                None => {
                    calls.push(TileCall { buffer, heap, ranges: Vec::new(), heap_tiles: Vec::new() });
                    calls.len() - 1
                }
            };
            let prev_end = calls[call].ranges.last().map(|&(t, n)| t + n);
            let same_heap_contig = matches!(target, Target::Null)
                || calls[call]
                    .ranges
                    .last()
                    .zip(calls[call].heap_tiles.last())
                    .is_some_and(|(&(_, n), &ht)| {
                        matches!(target, Target::Heap(p) if ht + n == heap_of(p).1)
                    });
            if prev_end == Some(tile) && same_heap_contig && let Some(range) = calls[call].ranges.last_mut() {
                range.1 += pages;
                continue;
            }
            calls[call].ranges.push((tile, pages));
            calls[call].heap_tiles.push(match target {
                Target::Heap(p) => heap_of(p).1,
                Target::Null => 0,
            });
        }
        calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn reserve(m: &mut Mirror, va: u64, pages: u32) {
        assert_eq!(m.apply(Change::Reserve { va, pages, on_demand: false }), None);
    }

    /// Maps and gives every page its tile at once, as its first copy does.
    fn map(m: &mut Mirror, va: u64, pages: u32, first_page: u32) -> Option<u64> {
        let refused = m.apply(Change::Map { va, pages, first_page });
        if refused.is_none() && m.translate(va, 1).is_some() {
            for k in 0..pages {
                m.give_tile(first_page + k);
            }
        }
        refused
    }

    #[test]
    fn a_span_of_bytes_covers_whole_pages() {
        assert_eq!(page_span(0x25801160000, 8_294_400), (0x25801160000, 127));
        assert_eq!(page_span(0x1_2345, 1), (0x1_0000, 1));
        assert_eq!(page_span(0x1_FFFF, 2), (0x1_0000, 2));
        assert_eq!(page_span(0x2_0000, 0x1_0000), (0x2_0000, 1));
        assert_eq!(page_span(0x2_0000, 0), (0x2_0000, 1));
    }

    #[test]
    fn a_span_touches_the_written_tiles_as_its_mirror_tiles_say() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 16);
        map(&mut m, 0x1000_0000, 3, 0);
        let tiles = m.mirror_tiles(0x1000_0000, 3).unwrap();
        let written: std::collections::HashSet<u32> = [tiles[2]].into();
        assert_eq!(m.touches(0x1000_0000, 3, &written), Some(true));
        assert_eq!(m.touches(0x1000_0000, 2, &written), Some(false), "the first two pages were not written");
        assert_eq!(m.touches(0x1000_0000, 4, &written), None, "page 3 is not mapped");
        assert_eq!(m.touches(0x1000_0000 + 1, 1, &written), None, "not on a page");
    }

    #[test]
    fn a_span_is_fresh_until_its_tile_mappings_are_drained() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 16);
        map(&mut m, 0x1000_0000, 2, 0);
        let p = m.translate(0x1000_0000, 2 * PAGE).unwrap();
        assert!(m.pending_in(p.buffer, p.offset, 2 * PAGE), "mapped but not applied");
        assert!(!m.drain_updates().is_empty());
        assert!(!m.pending_in(p.buffer, p.offset, 2 * PAGE), "applied");
        map(&mut m, 0x1000_0000 + 4 * PAGE, 1, 5);
        assert!(!m.pending_in(p.buffer, p.offset, 2 * PAGE), "a change after the span is not in it");
        assert!(m.pending_in(p.buffer, p.offset, 4 * PAGE + 1), "its last byte is");
    }

    #[test]
    fn a_mirror_tile_remembers_its_pool_page_across_aliases() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 4);
        m.apply(Change::Map { va: 0x1000_0000, pages: 1, first_page: 70 });
        m.apply(Change::Map { va: 0x1000_0000 + PAGE, pages: 1, first_page: 90 });
        assert_eq!(m.give_tile(70), 0);
        assert_eq!(m.give_tile(90), 1);
        assert_eq!(m.give_tile(70), 0, "an alias keeps its tile");
        assert_eq!((m.page_of_tile(0), m.page_of_tile(1), m.page_of_tile(2)), (Some(70), Some(90), None));
        assert_eq!((m.tile_of(70), m.tile_of(90)), (Some(0), Some(1)));
    }

    #[test]
    fn a_mirror_tile_is_a_tile_of_its_heap() {
        assert_eq!(heap_of(0), (0, 0));
        assert_eq!(heap_of(255), (0, 255));
        assert_eq!(heap_of(256), (1, 0));
        assert_eq!(heap_of(99775), (389, 191));
    }

    #[test]
    fn slots_are_given_on_the_first_map_by_first_fit_and_reused_after_a_release() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 16);
        reserve(&mut m, 0x2000_0000, 8);
        reserve(&mut m, 0x3000_0000, 4);
        assert!(m.extents().is_empty(), "a reservation that holds no page costs no window space");
        assert_eq!(m.generation(), 0);
        map(&mut m, 0x1000_0000, 1, 0);
        map(&mut m, 0x2000_0000, 1, 1);
        map(&mut m, 0x3000_0000, 1, 2);
        assert_eq!(m.extents(), [(0, 0, 16 * PAGE), (0, 16 * PAGE, 8 * PAGE), (0, 24 * PAGE, 4 * PAGE)], "packed one after the other");
        assert_eq!(m.generation(), 3);
        map(&mut m, 0x1000_0000 + PAGE, 1, 3);
        assert_eq!(m.generation(), 3, "a map inside a slot changes no address");
        m.apply(Change::Release { va: 0x2000_0000 });
        assert_eq!(m.generation(), 4);
        reserve(&mut m, 0x4000_0000, 6);
        map(&mut m, 0x4000_0000, 1, 9);
        assert_eq!(m.translate(0x4000_0000, 4).map(|p| p.offset), Some(16 * PAGE), "the freed slot is reused");
        reserve(&mut m, 0x5000_0000, 4);
        map(&mut m, 0x5000_0000, 1, 10);
        assert_eq!(m.translate(0x5000_0000, 4).map(|p| p.offset), Some(28 * PAGE), "the 2 pages left of the hole are too small, the end of the window serves");
    }

    #[test]
    fn a_full_window_makes_another_and_a_huge_reservation_gets_its_own() {
        let mut m = Mirror::new(8 * PAGE);
        reserve(&mut m, 0x1_0000_0000, 6);
        reserve(&mut m, 0x2_0000_0000, 6);
        reserve(&mut m, 0x3_0000_0000, 20);
        map(&mut m, 0x1_0000_0000, 1, 0);
        map(&mut m, 0x2_0000_0000, 1, 1);
        map(&mut m, 0x3_0000_0000, 1, 2);
        assert_eq!(m.buffer_bytes(), [8 * PAGE, 8 * PAGE, 20 * PAGE]);
        assert_eq!(m.extents(), [(0, 0, 6 * PAGE), (1, 0, 6 * PAGE), (2, 0, 20 * PAGE)]);
    }

    #[test]
    fn a_reservation_above_one_tebibyte_is_refused_once() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x10_0000_0000, 1);
        m.apply(Change::Reserve { va: 0x100_0000_0000, pages: ((MAX_SLOT_BYTES + PAGE) / PAGE) as u32, on_demand: false });
        assert_eq!(map(&mut m, 0x100_0000_0000, 1, 0), Some(MAX_SLOT_BYTES + PAGE));
        assert_eq!(map(&mut m, 0x100_0000_0000 + PAGE, 1, 1), None, "not logged again");
        assert!(m.translate(0x100_0000_0000, 4).is_none());
        assert!(m.drain_updates().is_empty());
    }

    #[test]
    fn mirror_tiles_follow_the_views_and_need_every_page_mapped() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 4);
        map(&mut m, 0x1000_0000, 2, 40);
        map(&mut m, 0x1000_0000 + 2 * PAGE, 1, 7);
        assert_eq!(m.mirror_tiles(0x1000_0000, 3), Some(vec![0, 1, 2]), "tiles in the order the pages were first mapped");
        assert_eq!(m.mirror_tiles(0x1000_0000 + PAGE, 2), Some(vec![1, 2]));
        assert_eq!(m.mirror_tiles(0x1000_0000, 4), None, "the last page is not mapped");
        assert_eq!(m.mirror_tiles(0x1000_0000, 5), None, "past the reservation");
        assert_eq!(m.mirror_tiles(0x1000_0000 + 0x20, 1), None, "not on a page");
        assert_eq!(m.mirror_tiles(0x2000_0000, 1), None, "no reservation");
        assert_eq!((m.tile_of(40), m.tile_of(7), m.tile_of(8)), (Some(0), Some(2), None));
    }

    #[test]
    fn scattered_pool_pages_share_one_heap() {
        // Pages of an arena on demand lie far apart in the pool: their tiles are next to each other.
        let mut m = Mirror::new(GIB);
        m.apply(Change::Reserve { va: 0x1000_0000, pages: 4, on_demand: true });
        map(&mut m, 0x1000_0000, 1, 5000);
        map(&mut m, 0x1000_0000 + 2 * PAGE, 1, 90_000);
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: Some(0), ranges: vec![(0, 1), (2, 1)], heap_tiles: vec![0, 1] }]);
        // The same pool page mapped again (an alias, or after a free) keeps its tile.
        map(&mut m, 0x1000_0000 + 3 * PAGE, 1, 5000);
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: Some(0), ranges: vec![(3, 1)], heap_tiles: vec![0] }]);
    }

    #[test]
    fn translate_gives_the_slot_address_and_the_room_after_the_span() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 4);
        reserve(&mut m, 0x2000_0000, 2);
        map(&mut m, 0x2000_0000, 1, 7);
        map(&mut m, 0x1000_0000, 1, 0);
        map(&mut m, 0x1000_0000 + 3 * PAGE, 1, 5);
        let near = m.translate(0x1000_0000 + 0x1234, 16).unwrap();
        assert_eq!(near, Placed { buffer: 0, offset: 2 * PAGE + 0x1234, left: 4 * PAGE - 0x1234 - 16 });
        assert_eq!(m.translate(0x1000_0000 + 4 * PAGE - 8, 8).map(|p| p.left), Some(0));
        assert_eq!(m.translate(0x1000_0000 + 4 * PAGE - 8, 9), None, "past the reservation");
        assert_eq!(m.translate(0x1000_0000 + PAGE, 4), None, "the first page is not mapped");
        let over_gap = m.translate(0x1000_0000 + PAGE - 8, 0x2_0000).unwrap();
        assert_eq!(over_gap.offset, 2 * PAGE + PAGE - 8, "a span over unmapped pages is contiguous in the window");
        assert_eq!(m.translate(0x1000, 4), None);
        assert_eq!(m.translate(0x1000_0000 + 5, 0).map(|p| p.offset), Some(2 * PAGE + 5), "a zero length counts as one byte");
    }

    #[test]
    fn tile_ranges_are_batched_for_each_heap_with_null_ranges_apart() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 8);
        // A page mapped and unmapped again gives this reservation its slot at the start of buffer 0 and takes
        // mirror tile 0. Then HEAP_PAGES - 3 pages of a commit on demand (in window buffer 1) take tiles 1 to HEAP_PAGES - 3.
        map(&mut m, 0x1000_0000 + 7 * PAGE, 1, 99_999);
        m.apply(Change::Unmap { va: 0x1000_0000 + 7 * PAGE, pages: 1 });
        m.apply(Change::Reserve { va: 0x4000_0000, pages: HEAP_PAGES - 3, on_demand: true });
        map(&mut m, 0x4000_0000, HEAP_PAGES - 3, 3000);
        m.drain_updates();
        // Two runs: the next four pages take the last two tiles of heap 0 (one range) and the first two
        // of heap 1, then a page mapped first long ago keeps its tile 41 (heap 0), one window tile on.
        map(&mut m, 0x1000_0000, 2, 100);
        map(&mut m, 0x1000_0000 + 2 * PAGE, 2, 102);
        map(&mut m, 0x1000_0000 + 5 * PAGE, 1, 3040);
        let calls = m.drain_updates();
        assert_eq!(
            calls,
            [
                TileCall { buffer: 0, heap: Some(0), ranges: vec![(0, 2), (5, 1)], heap_tiles: vec![HEAP_PAGES - 2, 41] },
                TileCall { buffer: 0, heap: Some(1), ranges: vec![(2, 2)], heap_tiles: vec![0] },
            ]
        );
        assert!(m.drain_updates().is_empty(), "drained");
        m.apply(Change::Unmap { va: 0x1000_0000 + PAGE, pages: 2 });
        let calls = m.drain_updates();
        assert_eq!(calls, [TileCall { buffer: 0, heap: None, ranges: vec![(1, 2)], heap_tiles: vec![0] }]);
        assert!(m.translate(0x1000_0000, 4).is_some() && m.translate(0x1000_0000 + PAGE, 4).is_none() && m.translate(0x1000_0000 + 3 * PAGE, 4).is_some());
    }

    #[test]
    fn the_last_change_of_a_tile_wins_and_a_remap_replaces_the_page() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 4);
        map(&mut m, 0x1000_0000, 2, 10);
        m.apply(Change::Unmap { va: 0x1000_0000, pages: 2 });
        map(&mut m, 0x1000_0000 + PAGE, 1, 99);
        assert_eq!(m.drain_updates(), [
            TileCall { buffer: 0, heap: None, ranges: vec![(0, 1)], heap_tiles: vec![0] },
            TileCall { buffer: 0, heap: Some(0), ranges: vec![(1, 1)], heap_tiles: vec![2] },
        ]);
        // Two views of one page: two tiles, one heap tile.
        map(&mut m, 0x1000_0000 + 3 * PAGE, 1, 99);
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: Some(0), ranges: vec![(3, 1)], heap_tiles: vec![2] }]);
    }

    #[test]
    fn a_map_over_a_view_cuts_it_and_a_release_shows_nothing() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 8);
        map(&mut m, 0x1000_0000, 6, 0);
        map(&mut m, 0x1000_0000 + 2 * PAGE, 2, 50);
        m.drain_updates();
        assert!(m.translate(0x1000_0000 + 5 * PAGE, 4).is_some());
        m.apply(Change::Unmap { va: 0x1000_0000 + 3 * PAGE, pages: 2 });
        m.drain_updates();
        assert!(m.translate(0x1000_0000 + 2 * PAGE, 4).is_some() && m.translate(0x1000_0000 + 3 * PAGE, 4).is_none() && m.translate(0x1000_0000 + 5 * PAGE, 4).is_some());
        m.apply(Change::Release { va: 0x1000_0000 });
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: None, ranges: vec![(0, 3), (5, 1)], heap_tiles: vec![0, 0] }]);
        assert!(m.translate(0x1000_0000, 4).is_none());
        assert!(m.extents().is_empty());
    }

    #[test]
    fn an_on_demand_commit_wants_its_unmapped_pages_asked_for() {
        let mut m = Mirror::new(GIB);
        m.apply(Change::Reserve { va: 0x1000_0000, pages: 8, on_demand: true });
        reserve(&mut m, 0x2000_0000, 4);
        assert!(m.wants_use(0x1000_0000 + 0x40, 16), "nothing of it is mapped yet");
        assert!(!m.wants_use(0x2000_0000, 16), "pages of a reservation the title maps itself are never asked for");
        assert!(!m.wants_use(0x3000_0000, 16), "no reservation");
        assert!(m.translate(0x1000_0000 + 0x40, 16).is_none());
        map(&mut m, 0x1000_0000, 2, 30);
        assert!(!m.wants_use(0x1000_0000 + 0x40, 2 * PAGE - 0x40), "both pages are there");
        assert!(m.wants_use(0x1000_0000 + 0x40, 2 * PAGE), "the span reaches the third page");
        map(&mut m, 0x1000_0000 + 2 * PAGE, 6, 40);
        assert!(!m.wants_use(0x1000_0000, 1 << 40), "every page is mapped, the length is clipped to the reservation");
        assert_eq!(m.translate(0x1000_0000 + 0x40, 16).map(|p| (p.buffer, p.offset)), Some((0, 0x40)));
        map(&mut m, 0x2000_0000, 1, 9);
        m.apply(Change::Reserve { va: 0x3000_0000, pages: 2, on_demand: true });
        map(&mut m, 0x3000_0000, 1, 50);
        assert_eq!(m.extents(), [(0, 0, 8 * PAGE), (1, 0, 4 * PAGE), (0, 8 * PAGE, 2 * PAGE)], "commits on demand share a window buffer of their own, packed");
    }

    #[test]
    fn a_page_without_bytes_shows_nothing_until_its_tile_and_then_every_view_of_it_shows_the_tile() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 4);
        reserve(&mut m, 0x2000_0000, 2);
        m.apply(Change::Map { va: 0x1000_0000, pages: 2, first_page: 70 });
        m.apply(Change::Map { va: 0x2000_0000 + PAGE, pages: 1, first_page: 71 });
        assert!(m.translate(0x1000_0000, 4).is_some(), "the span translates, its window tiles read zeros");
        assert_eq!(m.tiles_given(), 0, "no page has a tile");
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: None, ranges: vec![(0, 2), (5, 1)], heap_tiles: vec![0, 0] }]);
        assert_eq!(m.give_tile(71), 0);
        assert_eq!(m.give_tile(71), 0, "a tile is given once");
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: Some(0), ranges: vec![(1, 1), (5, 1)], heap_tiles: vec![0, 0] }], "both views of the page show it");
        m.apply(Change::Unmap { va: 0x2000_0000 + PAGE, pages: 1 });
        m.drain_updates();
        m.apply(Change::Map { va: 0x2000_0000, pages: 1, first_page: 71 });
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: Some(0), ranges: vec![(4, 1)], heap_tiles: vec![0] }], "a new view of a page with bytes shows its tile at once");
        assert_eq!(m.mirror_tiles(0x1000_0000, 2), Some(vec![1, 0]), "a buffer placed over the pages gives the missing tile");
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: Some(0), ranges: vec![(0, 1)], heap_tiles: vec![1] }]);
    }

    #[test]
    fn a_reset_forgets_everything_and_a_snapshot_builds_it_again() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 4);
        map(&mut m, 0x1000_0000, 2, 3);
        m.drain_updates();
        let before = m.generation();
        m.reset();
        assert!(m.translate(0x1000_0000, 4).is_none());
        assert!(m.generation() > before);
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: None, ranges: vec![(0, 2)], heap_tiles: vec![0] }]);
        reserve(&mut m, 0x1000_0000, 4);
        map(&mut m, 0x1000_0000, 2, 3);
        assert!(m.translate(0x1000_0000, 4).is_some());
    }

    #[test]
    fn read_fast_path_checks_tiles_without_allocating_and_misses_until_given() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 4);
        assert_eq!(m.apply(Change::Map { va: 0x1000_0000, pages: 2, first_page: 10 }), None);
        assert!(!m.has_tiles_for_span(0x1000_0000, PAGE * 2));
        assert_eq!(m.mirror_tiles_read(0x1000_0000, 2), None);
        m.give_span(0x1000_0000, PAGE);
        assert!(m.has_tiles_for_span(0x1000_0000, PAGE));
        assert!(!m.has_tiles_for_span(0x1000_0000, PAGE * 2));
        assert_eq!(m.mirror_tiles_read(0x1000_0000, 1), Some(vec![0]));
        assert_eq!(m.mirror_tiles_read(0x1000_0000, 2), None);
        assert_eq!(m.mirror_tiles(0x1000_0000, 2), Some(vec![0, 1]));
        assert!(m.has_tiles_for_span(0x1000_0000, PAGE * 2));
        assert_eq!(m.mirror_tiles_read(0x1000_0000, 2), Some(vec![0, 1]));
        assert!(!m.has_tiles_for_span(0x2000_0000, PAGE));
    }

    #[test]
    fn bulk_map_and_scatter_unmap_coalesce_pending_runs_across_heap_boundaries() {
        let mut m = Mirror::new(GIB);
        let pages = 99_776u32;
        reserve(&mut m, 0x2_0000_0000, pages);
        assert_eq!(m.apply(Change::Map { va: 0x2_0000_0000, pages, first_page: 0 }), None);
        assert_eq!(m.pending.len(), 1, "un-tiled bulk map must produce a single pending run");
        assert!(m.pending_in(0, 50_000 * PAGE, PAGE), "interior offset must still see pending run");
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: None, ranges: vec![(0, pages)], heap_tiles: vec![0] }]);
        assert!(!m.pending_in(0, 50_000 * PAGE, PAGE));

        // Give contiguous tiles across heap 0 -> heap 1 boundary and remap overlapping middle slice.
        for p in 0..300u32 {
            m.give_tile(p);
        }
        let calls = m.drain_updates();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], TileCall { buffer: 0, heap: Some(0), ranges: vec![(0, HEAP_PAGES)], heap_tiles: vec![0] });
        assert_eq!(calls[1], TileCall { buffer: 0, heap: Some(1), ranges: vec![(HEAP_PAGES, 300 - HEAP_PAGES)], heap_tiles: vec![0] });

        // Overwrite a middle slice with Null (unmap) and verify left/right split keeps exact tiles.
        m.apply(Change::Unmap { va: 0x2_0000_0000 + 200 * PAGE, pages: 60 });
        assert!(m.pending_in(0, 210 * PAGE, 4));
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: None, ranges: vec![(200, 60)], heap_tiles: vec![0] }]);
        assert_eq!(m.translate(0x2_0000_0000 + 199 * PAGE, 4).map(|p| p.offset), Some(199 * PAGE));
        assert!(m.translate(0x2_0000_0000 + 210 * PAGE, 4).is_none());
        assert_eq!(m.translate(0x2_0000_0000 + 260 * PAGE, 4).map(|p| p.offset), Some(260 * PAGE));
    }

    #[test]
    fn pages_in_heap_and_remap_heap_tiles_cover_dense_aliased_and_sparse_owners() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 300);
        reserve(&mut m, 0x2000_0000, 4);
        assert_eq!(m.apply(Change::Map { va: 0x1000_0000, pages: 260, first_page: 0 }), None);
        // Alias pool page 257 into the second reservation, plus a sparse page index in heap 1.
        assert_eq!(m.apply(Change::Map { va: 0x2000_0000, pages: 1, first_page: 257 }), None);
        let sparse_page = DENSE_OWNER_LIMIT as u32 + 17;
        assert_eq!(m.apply(Change::Map { va: 0x2000_0000 + PAGE, pages: 1, first_page: sparse_page }), None);
        for p in 0..260u32 {
            m.give_tile(p);
        }
        let sparse_tile = m.give_tile(sparse_page);
        assert_eq!(heap_of(sparse_tile), (1, 4));
        // Unmap page 256 so it is no longer owned by any window tile.
        assert_eq!(m.apply(Change::Unmap { va: 0x1000_0000 + 256 * PAGE, pages: 1 }), None);
        let _ = m.drain_updates();

        assert_eq!(m.pages_in_heap(0).len(), HEAP_PAGES as usize);
        assert_eq!(m.pages_in_heap(1), vec![256, 257, 258, 259, sparse_page]);
        assert!(m.pages_in_heap(99).is_empty());
        assert!(m.pages_in_heap(u32::MAX).is_empty());

        // Remapping heap 1 must re-queue 257..=259 in slot 0, the alias of 257 at tile 300,
        // and the sparse page at tile 301, while omitting unmapped page 256.
        assert_eq!(m.remap_heap_tiles(1), 5);
        assert_eq!(m.remap_heap_tiles(99), 0);
        assert_eq!(m.remap_heap_tiles(u32::MAX), 0);
        assert_eq!(
            m.drain_updates(),
            [TileCall {
                buffer: 0,
                heap: Some(1),
                ranges: vec![(257, 3), (300, 1), (301, 1)],
                heap_tiles: vec![1, 1, 4],
            }]
        );
    }

    #[test]
    fn checked_page_span_computes_sub_page_offsets_crossings_and_rejects_overflow() {
        assert_eq!(checked_page_span(0x1000_0000, 0), None);
        assert_eq!(checked_page_span(0x1000_0000, PAGE), Some((0x1000_0000, 0, 1)));
        assert_eq!(checked_page_span(0x1000_1800, 0x4000), Some((0x1000_0000, 0x1800, 1)));
        assert_eq!(checked_page_span(0x1000_1800, 0x20_0000), Some((0x1000_0000, 0x1800, 33)));
        assert_eq!(checked_page_span(0x1000_F900, 0x2000), Some((0x1000_0000, 0xF900, 2)));
        assert_eq!(checked_page_span(0x1000_B1B0, 16), Some((0x1000_0000, 0xB1B0, 1)));
        assert_eq!(checked_page_span(0x1000_FFFF, 1), Some((0x1000_0000, 0xFFFF, 1)));
        assert_eq!(checked_page_span(0x1000_FFFF, 2), Some((0x1000_0000, 0xFFFF, 2)));
        assert_eq!(checked_page_span(u64::MAX - 15, 32), None);
        assert_eq!(checked_page_span(0x1000_0000, (u64::from(u32::MAX) + 1) * PAGE), None);
    }

    #[test]
    fn mirror_tiles_span_handles_sub_page_alignments_and_straddling_spans() {
        let mut m = Mirror::new(GIB);
        reserve(&mut m, 0x1000_0000, 4);
        assert_eq!(m.apply(Change::Map { va: 0x1000_0000, pages: 3, first_page: 20 }), None);
        let _ = m.drain_updates();

        // Read-only lookup misses before tiles are assigned.
        assert_eq!(m.mirror_tiles_span_read(0x1000_1800, 0x4000), None);

        // Single-page sub-64KiB placement allocates tile 0 and queues the window update.
        let single = m.mirror_tiles_span(0x1000_1800, 0x4000).expect("single sub-page span");
        assert_eq!(single, PlacedTiles { aligned_va: 0x1000_0000, base_offset: 0x1800, tiles: vec![0] });
        assert_eq!(m.mirror_tiles_span_read(0x1000_1800, 0x4000), Some(single));
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: Some(0), ranges: vec![(0, 1)], heap_tiles: vec![0] }]);

        // Straddling sub-64KiB placement across pages 0..2 reuses tile 0 and allocates tiles 1 and 2.
        let straddle = m.mirror_tiles_span(0x1000_F900, PAGE + 0x2000).expect("straddling span");
        assert_eq!(straddle, PlacedTiles { aligned_va: 0x1000_0000, base_offset: 0xF900, tiles: vec![0, 1, 2] });
        assert_eq!(m.mirror_tiles_span_read(0x1000_F900, PAGE + 0x2000), Some(straddle));
        assert_eq!(m.drain_updates(), [TileCall { buffer: 0, heap: Some(0), ranges: vec![(1, 2)], heap_tiles: vec![1] }]);

        // 16-byte aligned span on page 2 succeeds with exact base_offset.
        let small = m.mirror_tiles_span(0x1002_B1B0, 0x400).expect("16B aligned span");
        assert_eq!(small, PlacedTiles { aligned_va: 0x1002_0000, base_offset: 0xB1B0, tiles: vec![2] });

        // Straddling into the unmapped 4th page, overflowing past the reservation, zero length,
        // or addressing an unreserved region all return None.
        assert_eq!(m.mirror_tiles_span(0x1002_F000, 0x2000), None, "reaches unmapped page 3");
        assert_eq!(m.mirror_tiles_span(0x1000_1800, 4 * PAGE), None, "crosses reservation end");
        assert_eq!(m.mirror_tiles_span(0x1000_1800, 0), None, "zero length");
        assert_eq!(m.mirror_tiles_span(0x2000_1800, 64), None, "unreserved");
    }
}
