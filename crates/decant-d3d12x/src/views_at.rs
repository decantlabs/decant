//! Lookups keyed by console GPU virtual address. Views made at an address (device slots 70 and 71,
//! `CreateSrvAt` and `CreateUavAt`) find the host resource placed there, and device slots 74 and 75
//! (`RegisterPagePoolX` and `UnregisterPagePoolX`) keep a small table of page pools. Resources
//! enter the address table when they are placed through slots 44 and 133.

use std::collections::BTreeMap;
#[cfg(windows)]
use std::sync::Mutex;

/// One host resource recorded at a console GPU virtual address (from slots 44/133).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaResource {
    pub va: u64,
    pub size: u64,
    pub host_res: usize,
    pub is_buffer: bool,
}

/// Host resources ordered by the console address they were placed at.
#[derive(Default)]
pub struct VaTable {
    by_start: BTreeMap<u64, VaResource>,
}

impl VaTable {
    const fn empty() -> VaTable {
        VaTable { by_start: BTreeMap::new() }
    }

    /// Notes a placement. An entry without an address or a resource says nothing and is dropped,
    /// a second placement at the same address replaces the first.
    pub fn record(&mut self, entry: VaResource) {
        if entry.va != 0 && entry.host_res != 0 {
            self.by_start.insert(entry.va, entry);
        }
    }

    /// The nearest entry that starts at or below `va`, with the distance from its start.
    fn below(&self, va: u64) -> Option<(&VaResource, u64)> {
        let (_, e) = self.by_start.range(..=va).next_back()?;
        Some((e, va - e.va))
    }

    /// Finds what a view at `va` should use, as `(host_res, byte_offset)`. A buffer view may land
    /// anywhere inside a buffer, a texture view only on the first byte of a texture. The two kinds
    /// never answer for each other: the PC runtime removes the device when a buffer view is made of
    /// a texture (seen as reason 0x887A0001 right after `CreateShaderResourceView` for a raw view
    /// at the address of a texture).
    pub fn lookup(&self, va: u64, want_buffer: bool) -> Option<(usize, u64)> {
        let (e, off) = self.below(va)?;
        let hit = if want_buffer { e.is_buffer && off < e.size } else { !e.is_buffer && off == 0 };
        hit.then_some((e.host_res, off))
    }

    /// The buffer that holds all of `[va, va + len)`, as `(host_res, byte_offset)`.
    pub fn lookup_span(&self, va: u64, len: u64) -> Option<(usize, u64)> {
        let (e, off) = self.below(va)?;
        let end = off.checked_add(len)?;
        (e.is_buffer && end <= e.size).then_some((e.host_res, off))
    }

    /// Drops every entry of `host_res` (the resource is gone).
    pub fn forget(&mut self, host_res: usize) {
        self.by_start.retain(|_, e| e.host_res != host_res);
    }
}

/// Element byte size for typed DXGI buffer SRV/UAV formats (used when `StructureByteStride == 0`
/// and `RAW` is not set).
pub fn format_bytes(fmt: u32) -> u32 {
    match fmt {
        // 16-byte (RGBA32)
        1..=4 => 16,
        // 12-byte (RGB32)
        5..=8 => 12,
        // 8-byte (RGBA16, RG32)
        9..=18 => 8,
        // 4-byte (R10G10B10A2, R11G11B10, RGBA8, RG16, R32, BGRA8, BGRX8)
        23..=43 | 87..=93 => 4,
        // 2-byte (RG8, R16, B5G6R5, B5G5R5A1, B4G4R4A4)
        48..=59 | 85 | 86 | 115 => 2,
        // 1-byte (R8, A8)
        60..=65 => 1,
        _ => 0,
    }
}

/// Whether DXGI format `fmt` is block compressed (BC1 to BC7 with their typeless and sRGB forms).
pub fn block_compressed(fmt: u32) -> bool {
    matches!(fmt, 70..=84 | 94..=99)
}

/// Environment variable that decides whether block compressed textures get a view texture too.
pub const VIEW_BC_ENV: &str = "DECANT_D3D12X_VIEW_BC";

/// Whether this value of [`VIEW_BC_ENV`] turns block compressed view textures on. They are **on by
/// default** (the cause of the device removal during the world load is fixed, a buffer
/// view at the address of such a texture no longer takes the texture) and only `0` turns them off.
/// `1`, which the earlier opt-in of this branch used, still means on.
pub fn view_bc_switch(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// Whether block compressed textures get a texture for their views at an address (on by default,
/// `DECANT_D3D12X_VIEW_BC=0` turns it off).
pub fn view_bc() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| view_bc_switch(std::env::var(VIEW_BC_ENV).ok().as_deref()))
}

/// The most textures [`ViewTextures`] makes in a run. Past it a view of a texture no resource holds
/// is skipped as before, so PC memory stays bounded whatever the title does.
pub const MAX_VIEW_TEXTURES: usize = 65536;

/// What [`ViewTextures::lookup`] found for an address and a description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewTexture {
    /// A texture made for this address with this description.
    Same(usize),
    /// A texture made for this address with another description, which the title has replaced.
    Changed(usize),
    /// None made for this address.
    New,
}

/// Textures the layer makes for views at an address that no resource holds (console device slot 70
/// with a texture description): the title keeps textures in its own memory and makes views of them
/// from the description alone (seen in a retail title: the artwork and the gradients of its loading view, and world
/// textures while the world loads, hundreds a second). The layer gives such an address a PC texture
/// of that description, which `detile` fills from the title's bytes, and keeps it for the next views
/// of the same address and description.
///
/// The textures stay while the heaps are under their memory cap (`DECANT_D3D12X_VIEW_TEX_MIB`, else the
/// dedicated video memory of the card less 2 GiB):
/// past it the least recently used ones that no view of their address asked for during
/// [`MIN_IDLE_PRESENTS`] presents and [`MIN_IDLE_MS`] ms are given back through `retire`. Until the
/// cap is reached nothing is released, as in the runs that lived 240 s and more. Releasing one when no host slot held
/// its view any more removed the device in three runs of three (a page
/// fault at GPU address 0 some 20 s after the first releases), and never in the two runs that kept
/// them: a list recorded before the release and submitted after it, or a copy of the view in a table
/// slot, still reads the texture. `view_heap` keeps each small one at 4 KiB.
#[derive(Default)]
pub struct ViewTextures {
    made: std::collections::HashMap<u64, ViewEntry>,
    /// Textures replaced by another description at their address: they may still be read.
    stale: Vec<ViewEntry>,
    count: usize,
    /// Bytes of the textures made and not released.
    live_bytes: u64,
    /// Monotonic counter bumped on every texture creation or reuse.
    use_seq: u64,
}

/// A texture made for a view at an address, with what the cache needs to give it back.
#[derive(Clone, Debug)]
pub struct ViewEntry {
    pub desc: [u8; 0x38],
    pub res: usize,
    /// The heap of `view_heap` it was placed in, `None` for a committed texture.
    pub heap: Option<usize>,
    pub bytes: u64,
    /// The present and the time (milliseconds) of its last view or reuse.
    pub last_present: u64,
    pub last_ms: u64,
    /// Monotonic sequence number of its last view or reuse.
    pub last_use: u64,
}

/// Memory the textures of views at an address may take when the card says nothing (MiB). A run
/// that took several GiB for tens of thousands of textures died of a lack of memory.
pub const DEFAULT_VIEW_TEX_MIB: u64 = 14 * 1024;
/// A texture is given back only when no view of its address was made for this many presents ...
pub const MIN_IDLE_PRESENTS: u64 = 600;
/// ... and this many milliseconds (a present is slow while the world loads).
pub const MIN_IDLE_MS: u64 = 120_000;
/// The most textures given back in one pass.
pub const MAX_PER_PASS: usize = 4096;

/// Memory kept free for everything else on the card (render targets, the title's own buffers, the
/// desktop) when the cap of the view textures follows the card (see [`cap_from`]).
pub const VIEW_TEX_RESERVE: u64 = 2 << 30;
/// The least the cap is, whatever the card says.
pub const VIEW_TEX_FLOOR: u64 = 512 << 20;

/// The cap of the cache of view textures in bytes (0: none). `over_ride` is `DECANT_D3D12X_VIEW_TEX_MIB`
/// in MiB and wins when it is a number. Else the whole dedicated video memory of the card (its local
/// budget when the dedicated figure is unknown) less [`VIEW_TEX_RESERVE`], at least [`VIEW_TEX_FLOOR`];
/// without any figure from the card, [`DEFAULT_VIEW_TEX_MIB`] (pure, for the tests).
pub fn cap_from(over_ride: Option<&str>, dedicated: u64, budget: u64) -> u64 {
    if let Some(mib) = over_ride.and_then(|v| v.trim().parse::<u64>().ok()) {
        return mib << 20;
    }
    let vram = match (dedicated, budget) {
        (0, 0) => return DEFAULT_VIEW_TEX_MIB << 20,
        (0, b) => b,
        (d, _) => d,
    };
    vram.saturating_sub(VIEW_TEX_RESERVE).max(VIEW_TEX_FLOOR)
}

/// What the cache of view textures may hold: `DECANT_D3D12X_VIEW_TEX_MIB` (MiB, 0 for no limit), else the
/// memory of the card less 2 GiB ([`cap_from`]), worked out once from the first device that asks.
/// The second figure is the card's dedicated memory and the third its budget, for the log.
#[cfg(windows)]
pub fn view_texture_cap(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> (u64, u64, u64) {
    static CAP: std::sync::OnceLock<(u64, u64, u64)> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let (dedicated, budget) = crate::view_heap::adapter_memory(device);
        let cap = cap_from(std::env::var("DECANT_D3D12X_VIEW_TEX_MIB").ok().as_deref(), dedicated, budget);
        crate::mem_owners::set_view_cap(cap);
        (cap, dedicated, budget)
    })
}

/// Default LRU eviction budget for placed and committed view textures (MiB).
pub const DEFAULT_VIEW_TEX_BUDGET_MIB: u64 = 3072;

/// The LRU eviction budget in bytes from `DECANT_D3D12X_VIEW_TEX_BUDGET_MIB` (0 disables eviction).
pub fn budget_from(over_ride: Option<&str>) -> u64 {
    over_ride.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(DEFAULT_VIEW_TEX_BUDGET_MIB) << 20
}

/// The LRU eviction budget of view textures in bytes (`DECANT_D3D12X_VIEW_TEX_BUDGET_MIB`, default
/// 3072 MiB, 0 disables eviction). Read once.
#[cfg(windows)]
pub fn view_texture_budget() -> u64 {
    static BUDGET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| budget_from(std::env::var("DECANT_D3D12X_VIEW_TEX_BUDGET_MIB").ok().as_deref()))
}

impl ViewEntry {
    /// A texture `res` of `bytes` made at `present` and `ms` for the description `desc`.
    pub fn new(desc: [u8; 0x38], res: usize, heap: Option<usize>, bytes: u64, present: u64, ms: u64) -> ViewEntry {
        ViewEntry { desc, res, heap, bytes, last_present: present, last_ms: ms, last_use: 0 }
    }
}

impl ViewTextures {
    pub fn lookup(&self, va: u64, desc: &[u8; 0x38]) -> ViewTexture {
        match self.made.get(&va) {
            Some(e) if &e.desc == desc => ViewTexture::Same(e.res),
            Some(e) => ViewTexture::Changed(e.res),
            None => ViewTexture::New,
        }
    }

    /// Looks up `va` and updates its last use timestamps in one lock acquisition on a hit.
    pub fn lookup_and_touch(&mut self, va: u64, desc: &[u8; 0x38], present: u64, ms: u64) -> ViewTexture {
        match self.made.get_mut(&va) {
            Some(e) if &e.desc == desc => {
                self.use_seq += 1;
                e.last_present = present;
                e.last_ms = ms;
                e.last_use = self.use_seq;
                ViewTexture::Same(e.res)
            }
            Some(e) => ViewTexture::Changed(e.res),
            None => ViewTexture::New,
        }
    }

    /// A new view of the texture made for `va`: it was used now.
    pub fn touch(&mut self, va: u64, present: u64, ms: u64) {
        if let Some(e) = self.made.get_mut(&va) {
            self.use_seq += 1;
            e.last_present = present;
            e.last_ms = ms;
            e.last_use = self.use_seq;
        }
    }

    /// Whether another texture may be made.
    pub fn room(&self) -> bool {
        self.made.len() + self.stale.len() < MAX_VIEW_TEXTURES
    }

    /// Notes the texture `res` made for `va` and `desc`, in place of the one made there before
    /// (which then waits in `stale`, a slot may still read it).
    pub fn note(&mut self, va: u64, mut e: ViewEntry) {
        let bytes = e.bytes;
        self.use_seq += 1;
        e.last_use = self.use_seq;
        if let Some(old) = self.made.insert(va, e) {
            self.stale.push(old);
        }
        self.count += 1;
        self.live_bytes += bytes;
    }

    /// Textures made so far.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Bytes of the textures made and not released.
    pub fn live_bytes(&self) -> u64 {
        self.live_bytes
    }

    /// Bytes the cache holds in memory: each heap with a live texture at `heap_bytes`, and the committed
    /// textures at their size. A heap with no live texture is free to be filled again and is not counted.
    pub fn held_bytes(&self, heap_bytes: u64) -> u64 {
        let mut heaps = std::collections::HashSet::new();
        let mut committed = 0u64;
        for e in self.made.values().chain(self.stale.iter()) {
            match e.heap {
                Some(h) => {
                    heaps.insert(h);
                }
                None => committed += e.bytes,
            }
        }
        heaps.len() as u64 * heap_bytes + committed
    }

    /// Takes out the textures to give back when the cache holds more than `cap` bytes ([`held_bytes`],
    /// `cap` 0: never). The unit is the heap: the least recently used heap **all** of whose textures are
    /// idle (no view of their address asked for during [`MIN_IDLE_PRESENTS`] presents and
    /// [`MIN_IDLE_MS`] ms at `present` and `ms`) and which no slot of a view heap holds a view of
    /// (`referenced`) goes whole, so that it can be filled again. Committed textures go one by one under
    /// the same two conditions. It stops once the cache is back at `cap`, at most [`MAX_PER_PASS`] textures. The
    /// caller releases them through `retire` (after the submissions in flight).
    pub fn collect(&mut self, cap: u64, heap_bytes: u64, present: u64, ms: u64, referenced: &std::collections::HashSet<usize>) -> Pass {
        let mut pass = Pass { held_before: self.held_bytes(heap_bytes), ..Pass::default() };
        pass.held_after = pass.held_before;
        if cap == 0 || pass.held_before <= cap {
            return pass;
        }
        let free = |e: &ViewEntry| {
            present.saturating_sub(e.last_present) >= MIN_IDLE_PRESENTS && ms.saturating_sub(e.last_ms) >= MIN_IDLE_MS && !referenced.contains(&e.res)
        };
        // per heap: whether every texture of it may go, and when one of them was last used
        let mut heaps: std::collections::BTreeMap<usize, (bool, u64)> = std::collections::BTreeMap::new();
        for e in self.made.values().chain(self.stale.iter()) {
            if let Some(h) = e.heap {
                let v = heaps.entry(h).or_insert((true, 0));
                v.0 &= free(e);
                v.1 = v.1.max(e.last_ms);
            }
        }
        let mut order: Vec<(u64, usize)> = heaps.iter().filter(|(_, v)| v.0).map(|(&h, v)| (v.1, h)).collect();
        order.sort_unstable();
        let target = cap;
        let mut gone_heaps = Vec::new();
        let mut taken = 0usize;
        for (_, h) in order {
            if pass.held_after <= target || taken >= MAX_PER_PASS {
                break;
            }
            let n = self.made.values().chain(self.stale.iter()).filter(|e| e.heap == Some(h)).count();
            taken += n;
            pass.held_after = pass.held_after.saturating_sub(heap_bytes);
            gone_heaps.push(h);
        }
        for h in &gone_heaps {
            let vas: Vec<u64> = self.made.iter().filter(|(_, e)| e.heap == Some(*h)).map(|(&va, _)| va).collect();
            for va in vas {
                if let Some(e) = self.made.remove(&va) {
                    self.live_bytes = self.live_bytes.saturating_sub(e.bytes);
                    pass.freed.push(e);
                }
            }
            let mut i = 0;
            while i < self.stale.len() {
                if self.stale[i].heap == Some(*h) {
                    let e = self.stale.remove(i);
                    self.live_bytes = self.live_bytes.saturating_sub(e.bytes);
                    pass.freed.push(e);
                } else {
                    i += 1;
                }
            }
        }
        pass.heaps_freed = gone_heaps.len();
        // committed textures one by one while still over the target
        if pass.held_after > target {
            let mut order: Vec<(u64, u64)> = self.made.iter().filter(|(_, e)| e.heap.is_none() && free(e)).map(|(&va, e)| (e.last_ms, va)).collect();
            order.sort_unstable();
            for (_, va) in order {
                if pass.held_after <= target || pass.freed.len() >= MAX_PER_PASS {
                    break;
                }
                if let Some(e) = self.made.remove(&va) {
                    self.live_bytes = self.live_bytes.saturating_sub(e.bytes);
                    pass.held_after = pass.held_after.saturating_sub(e.bytes);
                    pass.freed.push(e);
                }
            }
        }
        pass.heaps_in_use = self.held_bytes(heap_bytes).div_ceil(heap_bytes.max(1)) as usize;
        pass.stuck = pass.freed.is_empty() && pass.held_before > cap + cap / 2;
        pass
    }

    /// Evicts the least recently used idle and unreferenced textures when [`live_bytes`] exceeds
    /// `budget` (`budget` 0 disables eviction), enforcing [`MIN_IDLE_PRESENTS`] and [`MIN_IDLE_MS`]
    /// up to [`MAX_PER_PASS`] textures per pass.
    pub fn collect_budget(&mut self, budget: u64, heap_bytes: u64, present: u64, ms: u64, referenced: &std::collections::HashSet<usize>) -> Pass {
        let held_before = self.held_bytes(heap_bytes);
        let mut pass = Pass { held_before, held_after: held_before, ..Pass::default() };
        if budget == 0 || self.live_bytes <= budget {
            return pass;
        }
        let free = |e: &ViewEntry| {
            present.saturating_sub(e.last_present) >= MIN_IDLE_PRESENTS && ms.saturating_sub(e.last_ms) >= MIN_IDLE_MS && !referenced.contains(&e.res)
        };
        // `(last_use, kind, key)` where kind 0 is `stale` (res) and kind 1 is `made` (va).
        let mut order: Vec<(u64, u8, u64)> = Vec::new();
        for e in &self.stale {
            if free(e) {
                order.push((e.last_use, 0, e.res as u64));
            }
        }
        for (&va, e) in &self.made {
            if free(e) {
                order.push((e.last_use, 1, va));
            }
        }
        order.sort_unstable();

        let mut touched_heaps = std::collections::HashSet::new();
        for (_, kind, id) in order {
            if self.live_bytes <= budget || pass.freed.len() >= MAX_PER_PASS {
                break;
            }
            let removed = if kind == 0 {
                let res = id as usize;
                self.stale.iter().position(|e| e.res == res).map(|i| self.stale.swap_remove(i))
            } else {
                self.made.remove(&id)
            };
            if let Some(e) = removed {
                self.live_bytes = self.live_bytes.saturating_sub(e.bytes);
                if let Some(h) = e.heap {
                    touched_heaps.insert(h);
                }
                pass.freed.push(e);
            }
        }

        let live_heaps: std::collections::HashSet<usize> = self.made.values().chain(self.stale.iter()).filter_map(|e| e.heap).collect();
        pass.heaps_freed = touched_heaps.iter().filter(|h| !live_heaps.contains(h)).count();
        pass.held_after = self.held_bytes(heap_bytes);
        pass.heaps_in_use = pass.held_after.div_ceil(heap_bytes.max(1)) as usize;
        pass.stuck = pass.freed.is_empty() && self.live_bytes > budget;
        pass
    }
}

/// What a pass of [`ViewTextures::collect`] did.
#[derive(Debug, Default)]
pub struct Pass {
    /// The textures to give back.
    pub freed: Vec<ViewEntry>,
    pub heaps_freed: usize,
    pub heaps_in_use: usize,
    pub held_before: u64,
    pub held_after: u64,
    /// Nothing could go while the cache holds more than one and a half times its cap.
    pub stuck: bool,
}

#[cfg(windows)]
static VIEW_TEXTURES: Mutex<Option<ViewTextures>> = Mutex::new(None);

/// Runs `f` on the textures made for views at an address.
#[cfg(windows)]
pub fn with_view_textures<R>(f: impl FnOnce(&mut ViewTextures) -> R) -> R {
    f(VIEW_TEXTURES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(ViewTextures::default))
}

/// The page pools a title registered through device slots 74 and 75. A pool is a start address and
/// a page count, and its index in the table is what slot 74 returns.
#[derive(Default)]
pub struct PagePools {
    slots: Vec<Option<(u64, u32)>>,
}

impl PagePools {
    const fn empty() -> PagePools {
        PagePools { slots: Vec::new() }
    }

    /// Registers a pool and returns its index. The same pool twice gets the same index, and a
    /// slot freed by `unregister` is used before the table grows.
    pub fn register(&mut self, va: u64, pages: u32) -> u32 {
        let pool = Some((va, pages));
        let at = self.slots.iter().position(|s| *s == pool).or_else(|| self.slots.iter().position(Option::is_none));
        match at {
            Some(i) => {
                self.slots[i] = pool;
                i as u32
            }
            None => {
                self.slots.push(pool);
                (self.slots.len() - 1) as u32
            }
        }
    }

    /// Frees the slot of a pool. Returns whether the pool was registered.
    pub fn unregister(&mut self, va: u64, pages: u32) -> bool {
        match self.slots.iter_mut().find(|s| **s == Some((va, pages))) {
            Some(slot) => {
                *slot = None;
                true
            }
            None => false,
        }
    }
}

#[cfg(windows)]
static VA_TABLE: Mutex<VaTable> = Mutex::new(VaTable::empty());
#[cfg(windows)]
static PAGE_POOLS: Mutex<PagePools> = Mutex::new(PagePools::empty());

#[cfg(windows)]
pub fn record_resource(va: u64, size: u64, host_res: usize, is_buffer: bool) {
    VA_TABLE.lock().unwrap_or_else(|e| e.into_inner()).record(VaResource {
        va,
        size,
        host_res,
        is_buffer,
    });
}

#[cfg(windows)]
pub fn lookup_resource(va: u64, want_buffer: bool) -> Option<(usize, u64)> {
    VA_TABLE.lock().unwrap_or_else(|e| e.into_inner()).lookup(va, want_buffer)
}

/// The buffer holding the whole span `[va, va + len)`, `(host_res, byte_offset)`.
#[cfg(windows)]
pub fn lookup_span(va: u64, len: u64) -> Option<(usize, u64)> {
    VA_TABLE.lock().unwrap_or_else(|e| e.into_inner()).lookup_span(va, len)
}

/// The resource `host_res` was released: its addresses no longer lead anywhere.
#[cfg(windows)]
pub fn forget_resource(host_res: usize) {
    VA_TABLE.lock().unwrap_or_else(|e| e.into_inner()).forget(host_res);
}

#[cfg(windows)]
pub fn register_page_pool(va: u64, pages: u32) -> u32 {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner()).register(va, pages)
}

#[cfg(windows)]
pub fn unregister_page_pool(va: u64, pages: u32) -> bool {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner()).unregister(va, pages)
}

/// The GPU virtual address a console title sees for a resource: the PC address, or, where PC answers
/// 0 (textures), the address the resource was placed at (0 when it was not placed).
pub fn title_gpu_va(pc_va: u64, placed_va: u64) -> u64 {
    if pc_va == 0 { placed_va } else { pc_va }
}

/// What `Map` answers for a resource (see `views_at::map_answer`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapAnswer {
    /// PC's own mapping, as PC returned it.
    Pc,
    /// PC could not map it; the title address of a placed resource.
    Placed(u64),
    /// PC could not map it and the resource was not placed: PC's failure is returned.
    Failed,
}

/// The `Map` answer for a resource. `pc_ok` is whether PC's `Map` succeeded and `placed_va` the title
/// address of the placement (0 when not placed). A placed resource that PC cannot map is CPU-visible
/// at its title address on the console, so PC's failure is not passed on for it.
pub fn map_answer(pc_ok: bool, placed_va: u64) -> MapAnswer {
    match (pc_ok, placed_va) {
        (true, _) => MapAnswer::Pc,
        (false, 0) => MapAnswer::Failed,
        (false, va) => MapAnswer::Placed(va),
    }
}

#[cfg(test)]
mod tests {
    use super::{MapAnswer, PagePools, VaResource, VaTable, ViewEntry, ViewTexture, ViewTextures, format_bytes, map_answer, title_gpu_va};

    #[test]
    fn block_compressed_view_textures_are_on_unless_zero() {
        assert!(super::view_bc_switch(None) && super::view_bc_switch(Some("1")) && super::view_bc_switch(Some(" 1 ")) && super::view_bc_switch(Some("")));
        assert!(!super::view_bc_switch(Some("0")) && !super::view_bc_switch(Some(" 0 ")));
    }

    #[test]
    fn block_compressed_formats_are_bc1_to_bc7() {
        assert!(super::block_compressed(71) && super::block_compressed(98) && super::block_compressed(80));
        assert!(!super::block_compressed(10) && !super::block_compressed(28) && !super::block_compressed(61));
    }

    #[test]
    fn the_cap_follows_the_card_less_two_gib_unless_the_variable_says_otherwise() {
        use super::cap_from;
        const GIB: u64 = 1 << 30;
        assert_eq!(cap_from(None, 16 * GIB, 15 * GIB), 14 * GIB, "the dedicated memory less 2 GiB, whatever the budget");
        assert_eq!(cap_from(None, 16 * GIB, 0), 14 * GIB);
        assert_eq!(cap_from(None, 0, 10 * GIB), 8 * GIB);
        assert_eq!(cap_from(None, 2 * GIB, 2 * GIB), 512 << 20, "a small card: the floor");
        assert_eq!(cap_from(None, 0, 0), 14 * GIB, "no figure: 14 GiB, not the old 2048 MiB");
        assert_eq!(cap_from(Some("1024"), 16 * GIB, 16 * GIB), GIB, "the variable wins");
        assert_eq!(cap_from(Some(" 0 "), 16 * GIB, 16 * GIB), 0, "0 means no limit");
        assert_eq!(cap_from(Some("many"), 16 * GIB, 0), 14 * GIB, "not a number: ignored");
    }

    #[test]
    fn a_view_texture_is_kept_per_address_and_description() {
        let mut t = ViewTextures::default();
        let mut a = [0u8; 0x38];
        a[0] = 3;
        let mut b = a;
        b[0x10] = 1;
        assert_eq!(t.lookup(0x1000, &a), ViewTexture::New);
        t.note(0x1000, ViewEntry::new(a, 0xAA, None, 100, 0, 0));
        assert_eq!(t.lookup(0x1000, &a), ViewTexture::Same(0xAA), "the same address and description reuse it");
        assert_eq!(t.lookup(0x1000, &b), ViewTexture::Changed(0xAA), "another description at the address replaces it");
        assert_eq!(t.lookup(0x2000, &a), ViewTexture::New);
        t.note(0x1000, ViewEntry::new(b, 0xBB, None, 100, 0, 0));
        assert_eq!((t.lookup(0x1000, &b), t.len()), (ViewTexture::Same(0xBB), 2), "the replaced one is still counted");
        assert!(t.room());
    }

    fn desc3() -> [u8; 0x38] {
        let mut d = [0u8; 0x38];
        d[0] = 3;
        d
    }

    /// `n` textures of 100 bytes in each of the heaps 0..heaps, made at `present` 0 and ms 0 then used at `used(heap)`.
    fn cache(heaps: usize, per_heap: usize, used: impl Fn(usize) -> (u64, u64)) -> ViewTextures {
        let mut t = ViewTextures::default();
        for h in 0..heaps {
            for k in 0..per_heap {
                let (p, ms) = used(h);
                t.note(0x1_0000 * (h as u64 + 1) + 0x1000 * k as u64, ViewEntry::new(desc3(), 0x100 * (h + 1) + k, Some(h), 100, p, ms));
            }
        }
        t
    }

    #[test]
    fn with_three_full_heaps_and_a_cap_of_two_the_oldest_heap_goes_whole() {
        use super::{MIN_IDLE_MS, MIN_IDLE_PRESENTS};
        let mut t = cache(3, 4, |h| (h as u64, h as u64 * 10));
        let (p, ms) = (MIN_IDLE_PRESENTS + 100, MIN_IDLE_MS + 1000);
        assert_eq!(t.held_bytes(1000), 3000);
        let none = std::collections::HashSet::new();
        assert!(t.collect(3000, 1000, p, ms, &none).freed.is_empty(), "at the cap: nothing goes");
        assert!(t.collect(0, 1000, p, ms, &none).freed.is_empty(), "cap 0: no limit");
        let pass = t.collect(2000, 1000, p, ms, &none);
        let mut res: Vec<usize> = pass.freed.iter().map(|e| e.res).collect();
        res.sort_unstable();
        assert_eq!(res, vec![0x100, 0x101, 0x102, 0x103], "heap 0, the least recently used, all of it");
        assert_eq!((pass.heaps_freed, pass.heaps_in_use, t.held_bytes(1000)), (1, 2, 2000));
        assert!(!pass.stuck);
    }

    #[test]
    fn a_heap_goes_only_if_every_texture_in_it_is_idle_and_none_is_held_by_a_slot() {
        use super::{MIN_IDLE_MS, MIN_IDLE_PRESENTS};
        let mut t = cache(2, 3, |_| (0, 0));
        let (p, ms) = (MIN_IDLE_PRESENTS + 10, MIN_IDLE_MS + 10);
        // one texture of heap 0 is used just now (a view made of its address)
        t.touch(0x1_0000, p, ms);
        let none = std::collections::HashSet::new();
        let pass = t.collect(300, 1000, p, ms, &none);
        assert!(pass.freed.iter().all(|e| e.heap == Some(1)) && pass.heaps_freed == 1, "heap 0 has a used texture, heap 1 goes");
        // heap 1 is gone, heap 0 stays whatever the cap: its used texture keeps it
        let pass = t.collect(100, 1000, p, ms, &none);
        assert!(pass.freed.is_empty() && pass.stuck, "nothing can go and the cache holds more than one and a half times the cap");
        // a slot that still holds a texture of a heap keeps the heap
        let mut u = cache(2, 3, |_| (0, 0));
        let held: std::collections::HashSet<usize> = [0x200 + 1].into_iter().collect();
        let pass = u.collect(1000, 1000, p, ms, &held);
        assert!(pass.freed.iter().all(|e| e.heap == Some(0)) && pass.heaps_freed == 1, "the heap with a texture a slot holds stays");
    }

    #[test]
    fn a_texture_replaced_at_its_address_waits_as_stale_in_its_heap() {
        use super::{MIN_IDLE_MS, MIN_IDLE_PRESENTS};
        let mut t = ViewTextures::default();
        let (a, mut b) = (desc3(), desc3());
        b[0x10] = 9;
        t.note(0x1000, ViewEntry::new(a, 1, Some(0), 600, 0, 0));
        t.note(0x1000, ViewEntry::new(b, 2, Some(0), 600, 0, 0));
        assert_eq!(t.live_bytes(), 1200);
        let none = std::collections::HashSet::new();
        let pass = t.collect(1, 1000, MIN_IDLE_PRESENTS, MIN_IDLE_MS, &none);
        assert_eq!(pass.freed.len(), 2, "both textures of the heap go with it, the stale one too");
        assert_eq!(t.live_bytes(), 0);
    }

    #[test]
    fn a_committed_texture_goes_alone_when_idle_and_unheld() {
        use super::{MIN_IDLE_MS, MIN_IDLE_PRESENTS};
        let mut t = ViewTextures::default();
        t.note(0x1000, ViewEntry::new(desc3(), 1, None, 600, 0, 0));
        t.note(0x2000, ViewEntry::new(desc3(), 2, None, 600, 0, 0));
        let held: std::collections::HashSet<usize> = [1].into_iter().collect();
        let pass = t.collect(1000, 1000, MIN_IDLE_PRESENTS, MIN_IDLE_MS, &held);
        assert_eq!(pass.freed.iter().map(|e| e.res).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn map_falls_back_to_the_placed_address_only_for_placed_resources() {
        assert_eq!(map_answer(true, 0x1000), MapAnswer::Pc, "PC's mapping wins, placed or not");
        assert_eq!(map_answer(true, 0), MapAnswer::Pc);
        assert_eq!(map_answer(false, 0xDCF8_1C600), MapAnswer::Placed(0xDCF8_1C600));
        assert_eq!(map_answer(false, 0), MapAnswer::Failed, "not placed: the PC failure stands");
    }

    #[test]
    fn a_placed_texture_answers_its_title_address_and_other_resources_their_pc_address() {
        // a texture: PC gives 0, the title sees where it was placed
        assert_eq!(title_gpu_va(0, 0xDCF8_1C600), 0xDCF8_1C600);
        // a buffer or a texture PC does answer: PC wins, the placement is not used
        assert_eq!(title_gpu_va(0x1234_0000, 0xDCF8_1C600), 0x1234_0000);
        // not placed by the title: 0 stays 0
        assert_eq!(title_gpu_va(0, 0), 0);
    }

    #[test]
    fn va_table_matches_buffer_interior_and_texture_exact() {
        let mut t = VaTable::default();
        t.record(VaResource { va: 0xA5EF_0000, size: 0x0AE0_0000, host_res: 0x1111, is_buffer: true });
        t.record(VaResource { va: 0xB000_0000, size: 0x0010_0000, host_res: 0x2222, is_buffer: false });
        assert_eq!(t.lookup(0xA5EF_0000, true), Some((0x1111, 0)));
        assert_eq!(t.lookup(0xA5EF_0100, true), Some((0x1111, 0x100)));
        assert_eq!(t.lookup(0xB000_0000, false), Some((0x2222, 0)));
        assert_eq!(t.lookup(0xB000_0100, false), None);
        assert_eq!(t.lookup(0xB000_0000, true), None, "a buffer view of a texture's address finds no resource");
        assert_eq!(t.lookup(0xA5EF_0000, false), None, "a texture view of a buffer's address finds no resource");
        assert_eq!(t.lookup(0xD4B0_70000, true), None);
        assert_eq!(format_bytes(28), 4);
        assert_eq!(format_bytes(2), 16);
    }

    #[test]
    fn a_span_must_lie_inside_one_buffer_and_a_released_resource_is_forgotten() {
        let mut t = VaTable::default();
        t.record(VaResource { va: 0x1000, size: 0x100, host_res: 0xA, is_buffer: true });
        t.record(VaResource { va: 0x9000, size: 0x100, host_res: 0xB, is_buffer: false });
        assert_eq!(t.lookup_span(0x1010, 0x20), Some((0xA, 0x10)));
        assert_eq!(t.lookup_span(0x10E0, 0x20), Some((0xA, 0xE0)));
        assert_eq!(t.lookup_span(0x10E0, 0x21), None);
        assert_eq!(t.lookup_span(0x0FFF, 1), None);
        assert_eq!(t.lookup_span(0x9000, 4), None, "a texture is not a span of bytes");
        assert_eq!(t.lookup_span(0x1000, u64::MAX), None);
        t.forget(0xA);
        assert_eq!(t.lookup_span(0x1010, 4), None);
    }

    #[test]
    fn page_pools_deduplicate_and_reuse_slots() {
        let mut p = PagePools::default();
        assert_eq!(p.register(0xDD47_70000, 100000), 0);
        assert_eq!(p.register(0xDD47_70000, 100000), 0);
        assert_eq!(p.register(0xEE00_00000, 128), 1);
        assert!(p.unregister(0xDD47_70000, 100000));
        assert_eq!(p.register(0xFF00_00000, 64), 0);
    }

    #[test]
    fn lru_budget_evicts_oldest_unused_textures_and_reclaims_empty_heaps() {
        use super::{DEFAULT_VIEW_TEX_BUDGET_MIB, MIN_IDLE_MS, MIN_IDLE_PRESENTS, budget_from};
        assert_eq!(budget_from(None), DEFAULT_VIEW_TEX_BUDGET_MIB << 20);
        assert_eq!(budget_from(Some("0")), 0);
        assert_eq!(budget_from(Some(" 512 ")), 512 << 20);

        let mut t = ViewTextures::default();
        t.note(0x1000, ViewEntry::new(desc3(), 1, Some(0), 500, 0, 0));
        t.note(0x2000, ViewEntry::new(desc3(), 2, Some(0), 500, 0, 0));
        t.note(0x3000, ViewEntry::new(desc3(), 3, Some(1), 500, 0, 0));

        let held: std::collections::HashSet<usize> = [2].into_iter().collect();
        // Before MIN_IDLE_PRESENTS and MIN_IDLE_MS have passed, nothing is evicted.
        assert!(t.collect_budget(600, 1000, MIN_IDLE_PRESENTS - 1, MIN_IDLE_MS, &held).freed.is_empty());
        assert!(t.collect_budget(600, 1000, MIN_IDLE_PRESENTS, MIN_IDLE_MS - 1, &held).freed.is_empty());

        // Touch 0x1000 via lookup_and_touch at present 10, ms 10 so 0x1000 is newer and not idle yet.
        assert_eq!(t.lookup_and_touch(0x1000, &desc3(), 10, 10), ViewTexture::Same(1));
        let (p, ms) = (MIN_IDLE_PRESENTS, MIN_IDLE_MS);
        let pass = t.collect_budget(600, 1000, p, ms, &held);
        let freed: Vec<usize> = pass.freed.iter().map(|e| e.res).collect();
        assert_eq!(freed, vec![3], "only idle unreferenced 3 goes; touched 1 is still active");
        assert_eq!(pass.heaps_freed, 1, "heap 1 became empty and was reclaimed");
        assert_eq!(t.live_bytes(), 1000);

        // Once 0x1000 also reaches idle age, it is evicted to bring live_bytes <= 600.
        let pass2 = t.collect_budget(600, 1000, p + 10, ms + 10, &held);
        assert_eq!(pass2.freed.iter().map(|e| e.res).collect::<Vec<_>>(), vec![1]);
        assert_eq!(t.live_bytes(), 500);
        assert_eq!(t.held_bytes(1000), 1000);
        assert!(!pass2.stuck);

        // Budget 0 disables eviction.
        assert!(t.collect_budget(0, 1000, p + 10, ms + 10, &held).freed.is_empty());
    }
}
