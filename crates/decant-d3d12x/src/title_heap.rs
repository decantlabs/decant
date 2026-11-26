//! Memory for descriptor heaps and display targets taken from the title's own allocator, and the
//! translation of CPU descriptor handles the title writes (addresses in that memory) to the CPU
//! handles of the host descriptor heap that stands behind them.
//!
//! A title exports `__XMemAllocationHooks__`, a table of an allocate and a free function. When both
//! point into the main module, descriptor heap memory is requested through them so that the title
//! finds its heaps where its own allocator put them.
//!
//! The switch `DECANT_D3D12X_TITLE_HEAP=0` turns the adoption off (it is on by default).

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Mutex, OnceLock, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

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

/// `XALLOC_ATTRIBUTES` for descriptor heap memory: object type 4, 64 KiB pages, allocator 0x80
/// (the D3D runtime), graphics memory.
pub const ATTR_DESCRIPTOR_HEAP: u64 = 4 | (1 << 14) | (0x80 << 16) | (16 << 24) | (1 << 29);

/// `XALLOC_ATTRIBUTES` for display target memory: 4 MiB pages, allocator 0x80, graphics memory.
pub const ATTR_DISPLAY_TARGET: u64 = (2 << 14) | (0x80 << 16) | (16 << 24) | (1 << 29);

/// PC D3D12 Tier 2/3 hard cap for shader-visible CBV/SRV/UAV descriptor heaps (`1,000,000`).
/// A retail title asks for more than a million slots on the console. That fails on Windows PC D3D12 with
/// `E_INVALIDARG`, also on a Tier 3 RTX 3090. Such a heap is paged ([`Pager`]).
pub const PC_MAX_SHADER_VISIBLE_VIEW_DESCRIPTORS: u32 = 1_000_000;

/// One registered descriptor heap mapped to a range of title memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub title_base: usize,
    pub title_bytes: usize,
    pub host_cpu: usize,
    pub host_bytes: usize,
    pub host_heap: usize,
}

/// The registered heaps, ordered by the start of their title memory. The ranges never overlap.
#[derive(Default)]
pub struct Table {
    by_base: BTreeMap<usize, Entry>,
}

impl Table {
    const fn empty() -> Table {
        Table { by_base: BTreeMap::new() }
    }

    /// Registers `entry` unless its title range overlaps one already registered or the entry is
    /// empty. The error says why.
    pub fn insert(&mut self, entry: Entry) -> Result<(), &'static str> {
        let end = entry.title_base.checked_add(entry.title_bytes).ok_or("title range overflows address space")?;
        if entry.title_bytes == 0 || entry.host_cpu == 0 || entry.host_bytes == 0 {
            return Err("empty heap range or null host CPU start");
        }
        if let Some((_, before)) = self.by_base.range(..=entry.title_base).next_back()
            && before.title_base.saturating_add(before.title_bytes) > entry.title_base
        {
            return Err("range overlaps a previous registered heap");
        }
        if let Some((&next, _)) = self.by_base.range(entry.title_base..).next()
            && next < end
        {
            return Err("range overlaps a following registered heap");
        }
        self.by_base.insert(entry.title_base, entry);
        Ok(())
    }

    /// Unregisters the heap `host_heap` and returns its entry.
    pub fn remove_by_heap(&mut self, host_heap: usize) -> Option<Entry> {
        let base = self.by_base.values().find(|e| e.host_heap == host_heap)?.title_base;
        self.by_base.remove(&base)
    }

    /// The start of the title memory of `host_heap`, when it is registered.
    pub fn title_start_for_heap(&self, host_heap: usize) -> Option<usize> {
        self.by_base.values().find(|e| e.host_heap == host_heap).map(|e| e.title_base)
    }

    /// Translates a CPU descriptor handle in title memory to the host heap.
    /// * `Some(Some(host))`: `cpu` lies in a registered heap and in the part the host heap has.
    /// * `Some(None)`: `cpu` lies in a registered heap, past the host heap that was cut short.
    /// * `None`: `cpu` lies in no registered heap, so it is a host handle already.
    pub fn translate(&self, cpu: usize) -> Option<Option<usize>> {
        let (_, e) = self.by_base.range(..=cpu).next_back()?;
        let off = cpu - e.title_base;
        if off >= e.title_bytes {
            return None;
        }
        Some((off < e.host_bytes).then(|| e.host_cpu.wrapping_add(off)))
    }
}

static TABLE: RwLock<Table> = RwLock::new(Table::empty());
/// Lowest and highest address (the end is one past the last byte) of any registered heap, a lock free first test.
static BOUND_LO: AtomicUsize = AtomicUsize::new(usize::MAX);
static BOUND_HI: AtomicUsize = AtomicUsize::new(0);

#[cfg(windows)]
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DECANT_D3D12X_TITLE_HEAP").is_none_or(|v| v != "0"))
}

/// Translates one CPU descriptor handle. The answer is the handle itself when it is in no title
/// heap, the host handle when it is in one, and `None` when it falls in the part of a shader
/// visible heap that was cut off to fit the PC limit.
pub fn translate_cpu(cpu: usize) -> Option<usize> {
    if !(BOUND_LO.load(Ordering::Relaxed)..BOUND_HI.load(Ordering::Relaxed)).contains(&cpu) {
        return Some(cpu);
    }
    match TABLE.read().unwrap_or_else(|e| e.into_inner()).translate(cpu) {
        None => Some(cpu),
        Some(host) => host,
    }
}

/// The start of the title memory registered for `host_heap`, if any.
pub fn title_start_for_heap(host_heap: usize) -> Option<usize> {
    if BOUND_HI.load(Ordering::Relaxed) == 0 {
        return None;
    }
    TABLE.read().unwrap_or_else(|e| e.into_inner()).title_start_for_heap(host_heap)
}

/// The first word of a slot tag. On the console a descriptor is plain memory, and the title copies
/// the descriptors of its tables with its own memory copies from the slots its view calls wrote. The
/// layer writes views into the host heap only, so it leaves a tag in the title's slot instead: this
/// word, the host slot of the view with the number of the view in that slot (`slot_tags`) and both
/// xored. A copy of the title's carries the tag along, and before a draw `sync_tables` copies the host
/// view of each tagged slot into the host slot the shader reads.
const TAG: u64 = 0x5349_5850_5644_4553;

/// The bytes of a slot tag.
pub const TAG_BYTES: usize = 24;

/// The tag of a slot whose view `word` names (`slot_tags::made`).
pub fn tag_words(word: u64) -> [u64; 3] {
    [TAG, word, TAG ^ word]
}

/// The word of a tag (`slot_tags`), `None` when `words` is no tag.
pub fn tagged_word(words: [u64; 3]) -> Option<u64> {
    (words[0] == TAG && words[2] == TAG ^ words[1] && words[1] != 0).then_some(words[1])
}

/// Writes the tag of the view the layer made in host slot `host` into the title slot `cpu`, when
/// `cpu` is in a registered heap and is not the host slot itself. Called after the view was
/// written, so it also marks the slot of a paged heap for the next [`flush`].
#[allow(unsafe_code)]
pub fn tag_slot(cpu: usize, host: usize) {
    written(host, 1);
    // a slot that has a view now is no longer one the zero slot null view (`zero_null`) wrote
    crate::zero_null::forget(host, 1, 0);
    if cpu == host || cpu < BOUND_LO.load(Ordering::Relaxed) || cpu >= BOUND_HI.load(Ordering::Relaxed) {
        return;
    }
    if TABLE.read().unwrap_or_else(|e| e.into_inner()).translate(cpu).is_none() {
        return;
    }
    // SAFETY: `cpu` is inside a registered heap, whose title memory is committed and writable for
    // its whole range (checked when it was registered), and a slot holds at least the tag.
    unsafe { std::ptr::write_unaligned(cpu as *mut [u64; 3], tag_words(crate::slot_tags::made(host))) };
}

/// Copies the tags of `n` slots of `increment` bytes from the title slots at `src` to those at `dst`,
/// for a copy of descriptors through the API, so that a later copy of the title's finds them.
/// Called after the copy, so it also marks the destination slots of a paged heap for the next
/// [`flush`].
#[allow(unsafe_code)]
pub fn copy_tags(dst: usize, src: usize, n: u32, increment: u32) {
    if let Some(host) = translate_cpu(dst) {
        written(host, n);
        crate::zero_null::forget(host, n, increment);
        crate::slot_tags::overwritten(host, n, increment);
    }
    let lo = BOUND_LO.load(Ordering::Relaxed);
    let hi = BOUND_HI.load(Ordering::Relaxed);
    let bytes = n as usize * increment as usize;
    let inside = |a: usize| a >= lo && a.saturating_add(bytes) <= hi;
    if bytes == 0 || dst == src || !inside(dst) || !inside(src) {
        return;
    }
    let t = TABLE.read().unwrap_or_else(|e| e.into_inner());
    let whole = |a: usize| t.translate(a).is_some() && t.translate(a + bytes - 1).is_some();
    if whole(dst) && whole(src) {
        // SAFETY: both ranges are inside registered heaps (committed and writable title memory).
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, bytes) };
    }
}

/// A shader visible heap with title memory: where a table handle of it points to in the title's
/// memory and in the host heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visible {
    pub gpu_start: u64,
    pub increment: u32,
    pub host_cpu: usize,
    pub host_count: u32,
    pub title_base: usize,
    pub title_count: u32,
    /// The PC heap type (0 views, 1 samplers).
    pub heap_type: u32,
    /// The PC device, for the copies.
    pub device: usize,
    /// The PC heap.
    pub heap: usize,
    /// Host slots past `host_count` this heap's PC allocation also has, for a ring of tables
    /// moved for one draw ([`move_table`]); 0 when the heap has none (the extra slots were
    /// refused, or this is not an unpaged shader-visible view heap). Never handed to the title:
    /// every place that reads this heap's slots as the title's uses `host_count`, not this.
    pub ring_slots: u32,
    /// The next host slot of that ring a moved table takes.
    pub ring_next: u32,
}

static VISIBLE: RwLock<Vec<Visible>> = RwLock::new(Vec::new());
/// The tables in the ring of each [`Visible`] heap (keyed by its `gpu_start`): host slot of the
/// first, console slot of the first, slots, and the present at which each was written (as
/// `Paged::moved`). Apart from [`Visible`], which is `Copy` and copied by every lookup.
type RingSets = Vec<(u32, u32, u32, u64)>;
static RING_MOVED: Mutex<Vec<(u64, RingSets)>> = Mutex::new(Vec::new());
/// At least one registered [`Visible`] heap has a ring (`ring_slots > 0`). Sticky like
/// [`ANY_PAGED`]: a cheap check so [`move_table`] need not lock [`VISIBLE`] on the common run
/// that never allocates a ring.
static ANY_RING: AtomicBool = AtomicBool::new(false);

/// Registers a shader visible heap that has title memory.
pub fn visible_created(v: Visible) {
    if v.ring_slots > 0 {
        ANY_RING.store(true, Ordering::Release);
    }
    let mut g = VISIBLE.write().unwrap_or_else(|e| e.into_inner());
    g.retain(|o| o.gpu_start != v.gpu_start);
    g.push(v);
    RING_MOVED.lock().unwrap_or_else(|e| e.into_inner()).retain(|o| o.0 != v.gpu_start);
}

/// Forgets the shader visible heap of PC heap `heap`.
pub fn visible_released(heap: usize) {
    crate::zero_null::clear();
    let mut g = VISIBLE.write().unwrap_or_else(|e| e.into_inner());
    let gone: Vec<u64> = g.iter().filter(|o| o.heap == heap).map(|o| o.gpu_start).collect();
    g.retain(|o| o.heap != heap);
    RING_MOVED.lock().unwrap_or_else(|e| e.into_inner()).retain(|o| !gone.contains(&o.0));
}

/// The slots of a table that are looked at for tags: a console table names no length the layer can
/// trust (the translated shaders read it at the offsets of their console code), so a fixed window.
pub const TABLE_WINDOW: u32 = 64;

/// One tagged slot of a table's window: its host slot and the word of its tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableSlot {
    pub dst: usize,
    pub word: u64,
}

impl TableSlot {
    /// The host slot of the view the tag names, `dst` itself for a view the title made in the table
    /// itself, which needs no copy.
    pub fn src(&self) -> usize {
        crate::slot_tags::host_of(self.word)
    }
}

/// The views a table at GPU handle `handle` reads: every slot of its window whose title memory holds a
/// tag. `read` reads the tag words of a title slot. Pure apart from `read`, for the tests.
pub fn table_copies(v: &Visible, handle: u64, read: impl Fn(usize) -> [u64; 3]) -> Vec<TableSlot> {
    let inc = u64::from(v.increment);
    if inc == 0 || handle < v.gpu_start {
        return Vec::new();
    }
    let first = (handle - v.gpu_start) / inc;
    let end = (first + u64::from(TABLE_WINDOW)).min(u64::from(v.title_count.min(v.host_count)));
    let mut out = Vec::new();
    for slot in first..end {
        let off = slot as usize * v.increment as usize;
        if let Some(word) = tagged_word(read(v.title_base + off)) {
            out.push(TableSlot { dst: v.host_cpu + off, word });
        }
    }
    out
}

/// Host slots that hold a constant buffer view whose title address is not on 256 bytes, with that
/// address and the title's size. A PC view cannot start there, so the view in the slot starts on the
/// 256 bytes below, and a table that reads the slot gets a view of a copy of the bytes instead.
static UNALIGNED_CBV: RwLock<Option<FastMap<(u64, u32)>>> = RwLock::new(None);
static UNALIGNED_CBV_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Notes that host slot `host` holds a constant buffer view of the unaligned title range `cbv`, or no
/// such view (`None`).
pub fn note_cbv(host: usize, cbv: Option<(u64, u32)>) {
    if cbv.is_none() && UNALIGNED_CBV_COUNT.load(Ordering::Relaxed) == 0 {
        return;
    }
    let mut g = UNALIGNED_CBV.write().unwrap_or_else(|e| e.into_inner());
    match cbv {
        Some(c) => {
            let m = g.get_or_insert_with(FastMap::default);
            if m.insert(host, c).is_none() {
                UNALIGNED_CBV_COUNT.store(m.len(), Ordering::Relaxed);
            }
        }
        None => {
            if let Some(m) = g.as_mut()
                && m.remove(&host).is_some()
            {
                UNALIGNED_CBV_COUNT.store(m.len(), Ordering::Relaxed);
            }
        }
    }
}

/// The unaligned title range of the constant buffer view in host slot `host`.
pub fn unaligned_cbv(host: usize) -> Option<(u64, u32)> {
    if UNALIGNED_CBV_COUNT.load(Ordering::Relaxed) == 0 {
        return None;
    }
    UNALIGNED_CBV.read().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&host).copied()
}

/// The registered shader visible heap that holds GPU handle `handle`.
pub fn visible_for(handle: u64) -> Option<Visible> {
    let g = VISIBLE.read().unwrap_or_else(|e| e.into_inner());
    g.iter().find(|v| handle >= v.gpu_start && handle < v.gpu_start + u64::from(v.title_count) * u64::from(v.increment)).copied()
}

// ---- a shader visible view heap larger than the PC allows ----
//
// The title's view heap has more slots than a PC shader visible heap may have. Its slots live in a
// CPU only heap of the full size (the backing heap): the title's CPU handles translate to it at the
// console index, so every view, every copy through the API and every tag sync writes there. The PC
// shader visible heap is a cache of it in pages of [`PAGE`] slots.
//
// A console page gets a host page when the title first writes into it, so a shader that reads the
// heap at a console index it computed finds every page that holds descriptors through the page map
// of the bound heap (host index = map[i >> 12] + (i & 4095), a raw buffer of one `uint` per console
// page that [`page_map_address`] gives, [`UNMAPPED`] for a page without a host page; a heap within
// the PC limit has a map too, with page `p` at `p * PAGE`). A table gets a
// contiguous run of host pages for its whole extent when it is bound, and the host handle of that
// run is what the PC list and the user data mirror get. A console page may sit in several host
// runs, the map names one of them. Each write to the backing heap marks the slot dirty in every host
// page that holds it, and the dirty ranges are copied to the host heap before the next draw or
// dispatch.
//
// A page that some table starts in is a table page (the title's per frame ring). Its host pages
// are recycled: a host page that a command list bound stays until that list is reset or released
// (it may be executed later, or again), and after that for [`KEEP_PRESENTS`] presents, for the GPU
// to finish the frames in flight. A page only written and read by index keeps its host page.

/// Console slots per page of a paged heap.
pub const PAGE: u32 = 4096;
/// `PAGE` as a shift, for the shaders that read the page map.
pub const PAGE_SHIFT: u32 = 12;
const _: () = assert!(1 << PAGE_SHIFT == PAGE);
/// Presents a host page waits after its last use before it may hold another console page.
pub const KEEP_PRESENTS: u64 = 6;
/// The extent of a table whose root signature range is unbounded, or whose root signature the
/// layer does not know.
pub const UNBOUNDED_EXTENT: u32 = 65_536;
/// A page map entry of a console page that has no host page.
pub const UNMAPPED: u32 = u32::MAX;

const NONE: u32 = u32::MAX;

/// A copy of `count` descriptors from backing slot `from` to host slot `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fill {
    pub from: u32,
    pub to: u32,
    pub count: u32,
}

/// Where a table was placed in the host heap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapped {
    /// The host slot of the table's first descriptor.
    pub host: u32,
    /// The copies that fill the host pages taken for it.
    pub fills: Vec<Fill>,
    /// A host page was taken before [`KEEP_PRESENTS`] presents had passed since its last use,
    /// because no other was free.
    pub early: bool,
}

/// What a paged heap did so far, for the log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageStats {
    /// The highest console slot written.
    pub highest_written: Option<u32>,
    /// Console pages ever written.
    pub written_pages: u32,
    /// Of those, pages some table starts in, and pages only written.
    pub table_pages: u32,
    pub index_pages: u32,
    /// Host pages that hold a console page now, and at most so far.
    pub in_use: u32,
    pub peak: u32,
    /// Host pages taken from one console page for another.
    pub evictions: u64,
    /// GPU fence drains performed so unpinned host pages could be reused safely.
    pub drained: u64,
    /// Host pages taken before [`KEEP_PRESENTS`] presents had passed.
    pub early: u64,
    /// Writes and tables that found no host page.
    pub refused: u64,
    /// Host pages in all.
    pub host_pages: u32,
}

/// Outcome of a single attempt to place a table in [`Pager`].
enum MapAttempt {
    Mapped(Mapped),
    NeedsDrain,
    Full,
}

/// Which console page each host page holds. Pure, for the tests.
pub struct Pager {
    /// Console slots.
    count: u32,
    /// Per console page, the host pages that hold it.
    copies: Vec<Vec<u32>>,
    /// Per console page, the host page the page map names, or [`NONE`].
    canonical: Vec<u32>,
    /// Per console page, whether a table starts in or spans it.
    table: Vec<bool>,
    /// Per console page, whether it was ever written.
    ever: Vec<bool>,
    /// Per host page, its console page or [`NONE`].
    console_of: Vec<u32>,
    /// Per host page, the present count at its last use.
    used: Vec<u64>,
    /// Monotonic GPU-drain epoch stamped onto host pages as they are bound, unpinned, or updated.
    gpu_epoch: u64,
    /// Highest `gpu_epoch` proven complete on every GPU queue.
    completed_gpu_epoch: u64,
    /// Per host page, the `gpu_epoch` of its last GPU-visible use.
    used_epoch: Vec<u64>,
    /// Per host page, how many command lists bound it since their last reset.
    pins: Vec<u32>,
    /// Per host page, the range of its slots that differs from the backing heap.
    dirty: Vec<Option<(u32, u32)>>,
    /// The host pages each command list bound since its last reset.
    lists: FastMap<Vec<u32>>,
    /// The page map changed since it was last taken.
    map_changed: bool,
    /// Next host page index to check for a free (`NONE`) slot.
    free_hint: u32,
    /// Reusable scratch buffers for zero allocation flushes.
    flush_buf: Vec<Fill>,
    map_buf: Vec<u32>,
    stats: PageStats,
}

impl Pager {
    /// A pager for `count` console slots over `host_count` slots of a host heap.
    pub fn new(count: u32, host_count: u32) -> Pager {
        let host_pages = (host_count / PAGE) as usize;
        let pages = count.div_ceil(PAGE) as usize;
        Pager {
            count,
            copies: vec![Vec::new(); pages],
            canonical: vec![NONE; pages],
            table: vec![false; pages],
            ever: vec![false; pages],
            console_of: vec![NONE; host_pages],
            used: vec![0; host_pages],
            gpu_epoch: 1,
            completed_gpu_epoch: 0,
            used_epoch: vec![0; host_pages],
            pins: vec![0; host_pages],
            dirty: vec![None; host_pages],
            lists: FastMap::default(),
            map_changed: true,
            free_hint: 0,
            flush_buf: Vec::new(),
            map_buf: Vec::new(),
            stats: PageStats { host_pages: host_pages as u32, ..PageStats::default() },
        }
    }

    fn host_pages(&self) -> u32 {
        self.console_of.len() as u32
    }

    /// The console slots of page `c`.
    fn page_len(&self, c: u32) -> u32 {
        PAGE.min(self.count - c * PAGE)
    }

    /// Reserves the current GPU-use epoch for a fence drain and advances to a new epoch so any
    /// concurrent bind, write, or unpin during the wait is not treated as covered by this drain.
    pub fn begin_gpu_drain(&mut self) -> u64 {
        let epoch = self.gpu_epoch;
        self.gpu_epoch += 1;
        epoch
    }

    /// Marks every GPU submission up to `epoch` complete, allowing unpinned host pages last used
    /// at or before `epoch` to be recycled immediately without waiting for [`KEEP_PRESENTS`].
    pub fn mark_gpu_idle(&mut self, epoch: u64) {
        self.completed_gpu_epoch = self.completed_gpu_epoch.max(epoch);
        self.gpu_epoch = self.gpu_epoch.max(epoch + 1);
    }

    /// Whether host page `h` is the one the page map names for a page only read by index, which it
    /// keeps for good.
    fn kept(&self, h: u32) -> bool {
        let c = self.console_of[h as usize];
        c != NONE && !self.table[c as usize] && self.canonical[c as usize] == h
    }

    /// Whether host page `h` may take another console page at present `now`.
    fn cold(&self, h: u32, now: u64) -> bool {
        let i = h as usize;
        self.console_of[i] == NONE
            || (self.pins[i] == 0
                && !self.kept(h)
                && (self.used[i] + KEEP_PRESENTS <= now || self.used_epoch[i] <= self.completed_gpu_epoch))
    }

    /// The first of `n` host pages for console pages `c0..c0 + n`. A page of the run either holds
    /// its console page already or is taken. In this order: a run that starts at a host page of
    /// `c0` (a run grows at its end), a run after a host page of the console page before (the title
    /// fills its ring in order, so runs stay contiguous), the cold run preferring redundant
    /// duplicate copies before sole copies and age-cold pages before GPU-drained pages, then the
    /// unpinned run used longest ago (`true`: taken early).
    fn window(&self, c0: u32, n: u32, now: u64) -> Option<(u32, bool)> {
        let hp = self.host_pages();
        if n > hp {
            return None;
        }
        let holds = |h: u32, k: u32| self.console_of[(h + k) as usize] == c0 + k;
        let cold = |h: u32| self.cold(h, now);
        let fits = |h: u32| h + n <= hp && (0..n).all(|k| holds(h, k) || cold(h + k));
        let prev_next = c0.checked_sub(1).into_iter().flat_map(|prev| self.copies[prev as usize].iter().map(|&p| p + 1));
        if let Some(h) = self.copies[c0 as usize].iter().copied().chain(prev_next).find(|&h| fits(h)) {
            return Some((h, false));
        }
        if n == 1 && self.stats.in_use < hp {
            let start = (self.free_hint as usize).min(hp as usize);
            if let Some(rel) = self.console_of[start..].iter().position(|&c| c == NONE) {
                return Some(((start + rel) as u32, false));
            }
            if let Some(pos) = self.console_of[..start].iter().position(|&c| c == NONE) {
                return Some((pos as u32, false));
            }
        }
        let mut best_cold: Option<((u32, u32, bool, u64), u32)> = None;
        let mut best_early: Option<((u32, u32, u64), u32)> = None;
        for h in 0..=hp - n {
            let mut max_age = 0u64;
            let mut sole_evictions = 0u32;
            let mut canonical_hits = 0u32;
            let mut needs_epoch_cold = false;
            let mut all_cold = true;
            let mut all_unpinned = true;
            for k in 0..n {
                let idx = (h + k) as usize;
                let c = self.console_of[idx];
                if c == c0 + k || c == NONE {
                    continue;
                }
                if self.pins[idx] != 0 || self.kept(h + k) {
                    all_unpinned = false;
                    all_cold = false;
                    break;
                }
                let is_canonical = self.canonical[c as usize] == h + k;
                if is_canonical {
                    canonical_hits += 1;
                    if self.copies[c as usize].len() == 1 {
                        sole_evictions += 1;
                    }
                }
                let used = self.used[idx];
                let age_cold = used + KEEP_PRESENTS <= now;
                let epoch_cold = self.used_epoch[idx] <= self.completed_gpu_epoch;
                if !age_cold {
                    if epoch_cold {
                        needs_epoch_cold = true;
                    } else {
                        all_cold = false;
                    }
                }
                max_age = max_age.max(used + 1);
            }
            if all_cold {
                if max_age == 0 {
                    return Some((h, false));
                }
                let score = (sole_evictions, canonical_hits, needs_epoch_cold, max_age);
                if best_cold.is_none_or(|(best_score, _)| score < best_score) {
                    best_cold = Some((score, h));
                }
            } else if all_unpinned && best_cold.is_none() {
                let score = (sole_evictions, canonical_hits, max_age);
                if best_early.is_none_or(|(best_score, _)| score < best_score) {
                    best_early = Some((score, h));
                }
            }
        }
        if let Some((_, h)) = best_cold {
            return Some((h, false));
        }
        best_early.map(|(_, h)| (h, true))
    }

    /// Host page `h` holds console page `c` from now on.
    fn assign(&mut self, h: u32, c: u32) {
        let old = self.console_of[h as usize];
        if old == NONE {
            self.stats.in_use += 1;
            self.stats.peak = self.stats.peak.max(self.stats.in_use);
            let hp = self.host_pages();
            if hp > 0 {
                self.free_hint = (h + 1) % hp;
            }
        } else {
            self.stats.evictions += 1;
            let o = old as usize;
            self.copies[o].retain(|&x| x != h);
            if self.canonical[o] == h {
                self.canonical[o] = self.copies[o].first().copied().unwrap_or(NONE);
                self.map_changed = true;
            }
        }
        self.console_of[h as usize] = c;
        self.copies[c as usize].push(h);
        if self.canonical[c as usize] == NONE {
            self.canonical[c as usize] = h;
            self.map_changed = true;
        }
        self.dirty[h as usize] = None;
    }

    /// Places the table at console slot `slot`, which reads `extent` descriptors, for command list
    /// `list` at present `now`: in a host run that already holds its console pages in order, else
    /// in host pages taken for it. `None` past the console slots or when no run is free.
    pub fn map(&mut self, list: usize, slot: u32, extent: u32, now: u64) -> Option<Mapped> {
        match self.map_inner(list, slot, extent, now, true) {
            MapAttempt::Mapped(m) => Some(m),
            MapAttempt::Full | MapAttempt::NeedsDrain => None,
        }
    }

    fn map_inner(&mut self, list: usize, slot: u32, extent: u32, now: u64, allow_early: bool) -> MapAttempt {
        if slot >= self.count {
            return MapAttempt::Full;
        }
        let extent = extent.clamp(1, self.count - slot);
        let c0 = slot / PAGE;
        let n = (slot + extent - 1) / PAGE - c0 + 1;
        for k in 0..n {
            self.table[(c0 + k) as usize] = true;
        }
        let Some((h, early)) = self.window(c0, n, now) else {
            self.stats.refused += 1;
            return MapAttempt::Full;
        };
        if early && !allow_early {
            return MapAttempt::NeedsDrain;
        }
        self.stats.early += u64::from(early);
        let mut fills: Vec<Fill> = Vec::new();
        for k in 0..n {
            if self.console_of[(h + k) as usize] == c0 + k {
                continue;
            }
            self.assign(h + k, c0 + k);
            let from = (c0 + k) * PAGE;
            let count = self.page_len(c0 + k);
            match fills.last_mut() {
                Some(f) if f.from + f.count == from => f.count += count,
                _ => fills.push(Fill { from, to: (h + k) * PAGE, count }),
            }
        }
        let pinned = self.lists.entry(list).or_default();
        for k in h..h + n {
            self.used[k as usize] = now;
            self.used_epoch[k as usize] = self.gpu_epoch;
            if !pinned.contains(&k) {
                pinned.push(k);
                self.pins[k as usize] += 1;
            }
        }
        MapAttempt::Mapped(Mapped { host: h * PAGE + slot % PAGE, fills, early })
    }

    /// Command list `list` was reset or released at present `now`: the host pages it bound are free
    /// to go once [`KEEP_PRESENTS`] presents have passed or a GPU drain completes.
    pub fn release_list(&mut self, list: usize, now: u64) {
        let Some(pinned) = self.lists.get_mut(&list) else {
            return;
        };
        for h in pinned.drain(..) {
            self.pins[h as usize] -= 1;
            self.used[h as usize] = self.used[h as usize].max(now);
            self.used_epoch[h as usize] = self.used_epoch[h as usize].max(self.gpu_epoch);
        }
    }

    /// Whether [`written`](Self::written) for `first..first + count` would take a host page early.
    fn written_needs_drain(&self, first: u32, count: u32, now: u64) -> bool {
        let end = first.saturating_add(count).min(self.count);
        if first >= end {
            return false;
        }
        let mut s = first;
        let mut free_needed = 0u32;
        let free_avail = self.host_pages().saturating_sub(self.stats.in_use);
        while s < end {
            let c = s / PAGE;
            let stop = (c * PAGE + PAGE).min(end);
            s = stop;
            if self.copies[c as usize].is_empty() {
                free_needed += 1;
                if free_needed > free_avail && matches!(self.window(c, 1, now), Some((_, true))) {
                    return true;
                }
            }
        }
        false
    }

    /// Backing slots `first..first + count` were written at present `now`. A page that has no host
    /// page gets one (filled whole at the next flush), the host pages of the others are marked.
    /// Returns whether a host page has something to flush.
    pub fn written(&mut self, first: u32, count: u32, now: u64) -> bool {
        let end = first.saturating_add(count).min(self.count);
        if first >= end {
            return false;
        }
        self.stats.highest_written = self.stats.highest_written.max(Some(end - 1));
        let mut any = false;
        let mut s = first;
        while s < end {
            let c = s / PAGE;
            let stop = (c * PAGE + PAGE).min(end);
            let (lo, hi) = (s % PAGE, (stop - 1) % PAGE + 1);
            s = stop;
            if !self.ever[c as usize] {
                self.ever[c as usize] = true;
                self.stats.written_pages += 1;
            }
            if self.copies[c as usize].is_empty() {
                let Some((h, early)) = self.window(c, 1, now) else {
                    self.stats.refused += 1;
                    continue;
                };
                self.stats.early += u64::from(early);
                self.assign(h, c);
                self.used[h as usize] = now;
                self.used_epoch[h as usize] = self.gpu_epoch;
                self.dirty[h as usize] = Some((0, self.page_len(c)));
                any = true;
                continue;
            }
            for &h in &self.copies[c as usize] {
                let d = &mut self.dirty[h as usize];
                *d = Some(d.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
                if self.canonical[c as usize] == h {
                    self.used[h as usize] = now;
                    self.used_epoch[h as usize] = self.gpu_epoch;
                }
                any = true;
            }
        }
        any
    }

    /// The copies that bring the dirty slots of the host pages up to date.
    pub fn take_flush(&mut self) -> Vec<Fill> {
        let mut out = Vec::new();
        for (h, d) in self.dirty.iter_mut().enumerate() {
            if let Some((lo, hi)) = d.take() {
                let c = self.console_of[h];
                out.push(Fill { from: c * PAGE + lo, to: h as u32 * PAGE + lo, count: hi - lo });
            }
        }
        out
    }

    /// Collects dirty fills into the internal scratch buffer without allocating.
    fn take_flush_into(&mut self) -> Vec<Fill> {
        let mut out = std::mem::take(&mut self.flush_buf);
        out.clear();
        for (h, d) in self.dirty.iter_mut().enumerate() {
            if let Some((lo, hi)) = d.take() {
                let c = self.console_of[h];
                out.push(Fill { from: c * PAGE + lo, to: h as u32 * PAGE + lo, count: hi - lo });
            }
        }
        out
    }

    /// The page map, when it changed since the last call.
    pub fn take_map(&mut self) -> Option<Vec<u32>> {
        if !std::mem::take(&mut self.map_changed) {
            return None;
        }
        Some(self.canonical.iter().map(|&h| if h == NONE { UNMAPPED } else { h * PAGE }).collect())
    }

    /// Writes the updated page map into the internal scratch buffer when changed.
    fn take_map_into(&mut self) -> Option<Vec<u32>> {
        if !std::mem::take(&mut self.map_changed) {
            return None;
        }
        let mut out = std::mem::take(&mut self.map_buf);
        out.clear();
        out.extend(self.canonical.iter().map(|&h| if h == NONE { UNMAPPED } else { h * PAGE }));
        Some(out)
    }

    /// Whether [`take_map`](Self::take_map) has a new map.
    pub fn map_changed(&self) -> bool {
        self.map_changed
    }

    /// The console slot that host slot `host` holds.
    pub fn console_slot(&self, host: u32) -> Option<u32> {
        let c = *self.console_of.get((host / PAGE) as usize)?;
        let slot = c.checked_mul(PAGE)? + host % PAGE;
        (c != NONE && slot < self.count).then_some(slot)
    }

    /// What the pager did so far.
    pub fn stats(&self) -> PageStats {
        let table_pages = self.ever.iter().zip(&self.table).filter(|&(&e, &t)| e && t).count() as u32;
        PageStats { table_pages, index_pages: self.stats.written_pages - table_pages, ..self.stats }
    }
}

/// A paged shader visible view heap.
pub struct Paged {
    /// The GPU start of the PC heap, which the title also gets.
    pub gpu_start: u64,
    pub increment: u32,
    /// The CPU start of the PC shader visible heap.
    pub host_cpu: usize,
    /// The CPU start of the backing heap.
    pub backing_cpu: usize,
    /// The mapped memory of the page map buffer, 0 for none.
    pub map_cpu: usize,
    /// The PC device, the PC heap and the backing heap.
    pub device: usize,
    pub heap: usize,
    pub backing: usize,
    /// The diagnostic charge of the backing heap, removed at its physical release.
    pub(crate) backing_charge: Option<crate::mem_owners::Charge>,
    pub pager: Pager,
    /// The host slots of the PC heap, the last [`MOVED_SLOTS`] of them outside the pager's pages.
    pub host_count: u32,
    /// The next host slot of that tail a moved table takes.
    pub moved_next: u32,
    /// The tables in that tail now: host slot of the first, console slot of the first, slots, and the
    /// present at which each was written.
    pub moved: Vec<(u32, u32, u32, u64)>,
}

/// Host slots kept at the end of a paged heap for tables a draw reads from slots of their own
/// ([`move_table`]), used as a ring. A moved table must outlive the lists that read it, which may
/// run up to [`KEEP_PRESENTS`] presents later. With 1024 slots the ring came round within one or
/// two presents (measured on RADV for the unpaged heap, and on Windows as `came round` in the
/// menus of a title) and the text lost letters. 16384 slots (256
/// tables) last some 16 to 32 presents, and of a heap clamped to 1,000,000 slots they leave the
/// pager 240 pages of 4096 slots, above the peak the menus of a retail title use.
pub const MOVED_SLOTS: u32 = 16384;

/// The extra host slots a view heap that is not paged gets for its ring of moved tables: 1024
/// tables of [`TABLE_WINDOW`] slots. Measured on RADV: with 1024 slots two runs of two lost the
/// device with a GPU read at address 0; with 65,536 a round lasted 85 to 138 presents, above the
/// [`KEEP_PRESENTS`] a list in flight needs, and a run of 600 s had no device loss. The heap is
/// not clamped here, so the ring can be large.
pub const UNPAGED_MOVED_SLOTS: u32 = 65_536;

/// A table moved for one draw: where it is now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moved {
    /// The host GPU handle and CPU handle of its first slot.
    pub gpu: u64,
    pub cpu: usize,
    /// The ring of moved tables came round to its start: an older moved table was overwritten.
    pub wrapped: bool,
}

/// Whether a moved table written at host slots `at..at + n` would overwrite a table of `moved` that
/// was written less than [`KEEP_PRESENTS`] presents before `now`: a list recorded then may still be
/// waiting to run, and reads that table (pure, for the tests).
fn overwrites_young(moved: &[(u32, u32, u32, u64)], at: u32, n: u32, now: u64) -> bool {
    moved.iter().any(|&(h, _, k, t)| h < at + n && at < h + k && t + KEEP_PRESENTS > now)
}

/// Moved tables not made because the ring had no run free of tables younger than [`KEEP_PRESENTS`].
static RING_FULL: AtomicU64 = AtomicU64::new(0);

/// How many times the ring of moved tables was full (see [`move_table`]).
pub fn ring_full() -> u64 {
    RING_FULL.load(Ordering::Relaxed)
}

/// Where in the tail `lo..hi` of a heap a moved table of `n` slots goes when the next free slot
/// is `next`: the start of its run and whether the ring came round (pure, for the tests).
fn moved_run(lo: u32, hi: u32, next: u32, n: u32) -> Option<(u32, bool)> {
    if n == 0 || hi.saturating_sub(lo) < n {
        return None;
    }
    let at = next.max(lo);
    if at + n > hi { Some((lo, true)) } else { Some((at, false)) }
}

/// Copies the `n` slots of the table at console handle `console` from the backing heap of its paged
/// heap into a run of host slots of their own (the tail of [`MOVED_SLOTS`] past the pages, used as a
/// ring), for a draw that reads the table while a slot of it differs from what an earlier draw of
/// the same list reads. On an unpaged shader-visible view heap (vkd3d-proton: always, since it
/// never refuses a heap and so never pages one) the same ring is taken from the tail of
/// [`UNPAGED_MOVED_SLOTS`] extra host slots the heap's own PC allocation was given past its
/// title-visible `host_count`, when it has one, and the table's slots are copied from the heap's own
/// host slots (there is no separate backing heap off the paged path). `None` when `console` is in
/// neither a paged heap nor a heap with a ring, and when the next run of the ring still holds a
/// table written less than [`KEEP_PRESENTS`] presents ago: that table is never overwritten (a list
/// that reads it may not have run yet), the caller then copies into the shared slots.
pub fn move_table(console: u64, n: u32, copy: CopyFn) -> Option<Moved> {
    if ANY_PAGED.load(Ordering::Acquire) {
        let mut g = PAGED.write().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = g.iter_mut().find(|p| {
            let (lo, hi) = p.console_range();
            console >= lo && console < hi
        }) {
            let inc = p.increment as usize;
            let slot = ((console - p.gpu_start) / inc as u64) as u32;
            let n = n.min(p.pager.count - slot);
            let lo = p.pager.host_pages() * PAGE;
            let (at, wrapped) = moved_run(lo, p.host_count, p.moved_next, n)?;
            let now = PRESENTS.load(Ordering::Relaxed);
            if overwrites_young(&p.moved, at, n, now) {
                RING_FULL.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            p.moved_next = at + n;
            p.moved.retain(|&(h, _, k, _)| h + k <= at || h >= at + n);
            p.moved.push((at, slot, n, now));
            copy(p.device, p.host_cpu + at as usize * inc, p.backing_cpu + slot as usize * inc, n);
            return Some(Moved { gpu: p.gpu_start + u64::from(at) * inc as u64, cpu: p.host_cpu + at as usize * inc, wrapped });
        }
    }
    if ANY_RING.load(Ordering::Acquire) {
        let mut g = VISIBLE.write().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = g.iter_mut().find(|v| {
            v.ring_slots > 0 && console >= v.gpu_start && console < v.gpu_start + u64::from(v.title_count) * u64::from(v.increment)
        }) {
            let inc = v.increment as usize;
            let slot = ((console - v.gpu_start) / inc as u64) as u32;
            let n = n.min(v.title_count - slot);
            let (at, wrapped) = moved_run(v.host_count, v.host_count + v.ring_slots, v.ring_next, n)?;
            let now = PRESENTS.load(Ordering::Relaxed);
            let mut rm = RING_MOVED.lock().unwrap_or_else(|e| e.into_inner());
            let at_gpu = v.gpu_start;
            let i = rm.iter().position(|o| o.0 == at_gpu).unwrap_or_else(|| {
                rm.push((at_gpu, Vec::new()));
                rm.len() - 1
            });
            let moved = &mut rm[i].1;
            if overwrites_young(moved, at, n, now) {
                RING_FULL.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            v.ring_next = at + n;
            moved.retain(|&(h, _, k, _)| h + k <= at || h >= at + n);
            moved.push((at, slot, n, now));
            copy(v.device, v.host_cpu + at as usize * inc, v.host_cpu + slot as usize * inc, n);
            return Some(Moved { gpu: v.gpu_start + u64::from(at) * inc as u64, cpu: v.host_cpu + at as usize * inc, wrapped });
        }
    }
    None
}

impl Paged {
    fn console_range(&self) -> (u64, u64) {
        (self.gpu_start, self.gpu_start + u64::from(self.pager.count) * u64::from(self.increment))
    }

    /// The console slot that host slot `host` holds: through the pager, or as a slot of a table
    /// moved into the tail.
    fn console_slot(&self, host: u32) -> Option<u32> {
        self.pager.console_slot(host).or_else(|| self.moved.iter().find(|&&(h, _, n, _)| host >= h && host < h + n).map(|&(h, slot, _, _)| slot + (host - h)))
    }
}

static PAGED: RwLock<Vec<Paged>> = RwLock::new(Vec::new());
static ANY_PAGED: AtomicBool = AtomicBool::new(false);
static DIRTY: AtomicBool = AtomicBool::new(false);
static PRESENTS: AtomicU64 = AtomicU64::new(0);

/// A copy of descriptors on the PC: (device, destination CPU handle, source CPU handle, count).
pub type CopyFn<'a> = &'a mut dyn FnMut(usize, usize, usize, u32);
/// Writes a page map into the mapped page map buffer at the address given.
pub type MapFn<'a> = &'a mut dyn FnMut(usize, &[u32]);

fn run_fill(p: &Paged, f: Fill, copy: CopyFn) {
    let inc = p.increment as usize;
    copy(p.device, p.host_cpu + f.to as usize * inc, p.backing_cpu + f.from as usize * inc, f.count);
}

/// Registers a paged heap.
pub fn paged_created(p: Paged) {
    let mut g = PAGED.write().unwrap_or_else(|e| e.into_inner());
    g.retain(|o| o.heap != p.heap);
    g.push(p);
    ANY_PAGED.store(true, Ordering::Release);
    DIRTY.store(true, Ordering::Release);
}

/// Forgets the paged heap of PC heap `heap`. Returns its backing heap, for the caller to release.
pub(crate) fn paged_released(heap: usize) -> Option<(usize, Option<crate::mem_owners::Charge>)> {
    if !ANY_PAGED.load(Ordering::Acquire) {
        return None;
    }
    let mut g = PAGED.write().unwrap_or_else(|e| e.into_inner());
    let at = g.iter().position(|p| p.heap == heap)?;
    let p = g.remove(at);
    Some((p.backing, p.backing_charge))
}

/// The page map buffer of a shader visible view heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageMap {
    /// The PC heap and its GPU start.
    pub heap: usize,
    pub gpu_start: u64,
    /// The GPU address of the map and its buffer object.
    pub address: u64,
    pub buffer: usize,
    /// The diagnostic charge of the upload buffer, removed at its physical release.
    pub(crate) charge: Option<crate::mem_owners::Charge>,
}

static PAGE_MAPS: RwLock<Vec<PageMap>> = RwLock::new(Vec::new());
static ANY_PAGE_MAPS: AtomicBool = AtomicBool::new(false);

/// The page map of a heap of `count` slots that is not paged: page `p` at host index `p * PAGE`.
pub fn identity_map(count: u32) -> Vec<u32> {
    (0..count.div_ceil(PAGE)).map(|p| p * PAGE).collect()
}

/// The entries of the page map of a paged heap before any page has a host page.
pub fn empty_map(count: u32) -> Vec<u32> {
    vec![UNMAPPED; count.div_ceil(PAGE) as usize]
}

/// Registers the page map of a shader visible view heap.
pub fn page_map_created(m: PageMap) {
    let mut g = PAGE_MAPS.write().unwrap_or_else(|e| e.into_inner());
    g.retain(|o| o.heap != m.heap);
    g.push(m);
    ANY_PAGE_MAPS.store(true, Ordering::Release);
}

/// Forgets the page map of PC heap `heap`. Returns its buffer, for the caller to release.
pub(crate) fn page_map_released(heap: usize) -> Option<(usize, Option<crate::mem_owners::Charge>)> {
    if !ANY_PAGE_MAPS.load(Ordering::Acquire) {
        return None;
    }
    let mut g = PAGE_MAPS.write().unwrap_or_else(|e| e.into_inner());
    let at = g.iter().position(|m| m.heap == heap)?;
    let m = g.remove(at);
    Some((m.buffer, m.charge))
}

/// The GPU address of the page map of the shader visible view heap that starts at GPU handle
/// `gpu_start`, for the root parameter a directly translated shader reads it from.
pub fn page_map_address(gpu_start: u64) -> Option<u64> {
    if !ANY_PAGE_MAPS.load(Ordering::Acquire) {
        return None;
    }
    PAGE_MAPS.read().unwrap_or_else(|e| e.into_inner()).iter().find(|m| m.gpu_start == gpu_start).map(|m| m.address)
}

/// A present happened. Returns the present count.
pub fn present() -> u64 {
    PRESENTS.fetch_add(1, Ordering::Relaxed) + 1
}

/// The present count so far.
pub fn presents() -> u64 {
    PRESENTS.load(Ordering::Relaxed)
}

/// What each paged heap did so far.
pub fn stats() -> Vec<PageStats> {
    if !ANY_PAGED.load(Ordering::Acquire) {
        return Vec::new();
    }
    PAGED.read().unwrap_or_else(|e| e.into_inner()).iter().map(|p| p.pager.stats()).collect()
}

/// Command list `list` was reset or released.
pub fn list_reset(list: usize) {
    if !ANY_PAGED.load(Ordering::Acquire) {
        return;
    }
    let now = PRESENTS.load(Ordering::Relaxed);
    for p in PAGED.write().unwrap_or_else(|e| e.into_inner()).iter_mut() {
        p.pager.release_list(list, now);
    }
}

/// Where a table handle of the title goes on the PC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableAt {
    /// The handle is not in a paged heap and stays as it is.
    Same,
    /// The host handle, and whether a host page was taken early.
    Host(u64, bool),
    /// No host run is free for it.
    Full,
}

/// Whether the title's GPU handle `handle` is in a paged heap.
pub fn paged(handle: u64) -> bool {
    ANY_PAGED.load(Ordering::Acquire)
        && PAGED.read().unwrap_or_else(|e| e.into_inner()).iter().any(|p| {
            let (lo, hi) = p.console_range();
            handle >= lo && handle < hi
        })
}

/// Maximum time [`table`] or [`written`] waits for already-submitted GPU work before falling back
/// to an early host-page eviction.
#[cfg(windows)]
const GPU_DRAIN_WAIT_MS: u32 = 200;

fn drain_submitted_gpu() -> bool {
    #[cfg(windows)]
    {
        crate::retire::wait_submitted_idle(GPU_DRAIN_WAIT_MS)
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// Places the table at the title's GPU handle `handle`, which reads `extent` descriptors, for
/// command list `list`, filling host pages taken for it through `copy`.
pub fn table(list: usize, handle: u64, extent: u32, copy: CopyFn) -> TableAt {
    table_with_drain(list, handle, extent, copy, &mut drain_submitted_gpu)
}

fn table_with_drain(list: usize, handle: u64, extent: u32, copy: CopyFn, drain: &mut dyn FnMut() -> bool) -> TableAt {
    if !ANY_PAGED.load(Ordering::Acquire) {
        return TableAt::Same;
    }
    let drain_epochs = {
        let now = PRESENTS.load(Ordering::Relaxed);
        let mut g = PAGED.write().unwrap_or_else(|e| e.into_inner());
        let Some(p) = g.iter_mut().find(|p| {
            let (lo, hi) = p.console_range();
            handle >= lo && handle < hi
        }) else {
            return TableAt::Same;
        };
        let inc = u64::from(p.increment);
        let slot = ((handle - p.gpu_start) / inc) as u32;
        match p.pager.map_inner(list, slot, extent, now, false) {
            MapAttempt::Mapped(m) => {
                for f in m.fills {
                    run_fill(p, f, copy);
                }
                if p.pager.map_changed() {
                    DIRTY.store(true, Ordering::Release);
                }
                return TableAt::Host(p.gpu_start + u64::from(m.host) * inc + (handle - p.gpu_start) % inc, m.early);
            }
            MapAttempt::Full => return TableAt::Full,
            MapAttempt::NeedsDrain => g.iter_mut().map(|ph| (ph.heap, ph.pager.begin_gpu_drain())).collect::<Vec<_>>(),
        }
    };
    let idle = drain();
    let now = PRESENTS.load(Ordering::Relaxed);
    let mut g = PAGED.write().unwrap_or_else(|e| e.into_inner());
    if idle {
        for (heap_id, epoch) in drain_epochs {
            if let Some(ph) = g.iter_mut().find(|ph| ph.heap == heap_id) {
                ph.pager.mark_gpu_idle(epoch);
                ph.pager.stats.drained += 1;
            }
        }
    }
    let Some(p) = g.iter_mut().find(|p| {
        let (lo, hi) = p.console_range();
        handle >= lo && handle < hi
    }) else {
        return TableAt::Same;
    };
    let inc = u64::from(p.increment);
    let slot = ((handle - p.gpu_start) / inc) as u32;
    match p.pager.map_inner(list, slot, extent, now, true) {
        MapAttempt::Mapped(m) => {
            for f in m.fills {
                run_fill(p, f, copy);
            }
            if p.pager.map_changed() {
                DIRTY.store(true, Ordering::Release);
            }
            TableAt::Host(p.gpu_start + u64::from(m.host) * inc + (handle - p.gpu_start) % inc, m.early)
        }
        MapAttempt::Full | MapAttempt::NeedsDrain => TableAt::Full,
    }
}

/// The title's GPU handle of a host handle that [`table`] gave, the handle itself outside paged
/// heaps, `None` for a host slot of a paged heap that holds no console page.
pub fn console_handle(handle: u64) -> Option<u64> {
    if !ANY_PAGED.load(Ordering::Acquire) {
        return Some(handle);
    }
    let g = PAGED.read().unwrap_or_else(|e| e.into_inner());
    let Some(p) = g.iter().find(|p| handle >= p.gpu_start && handle < p.gpu_start + u64::from(p.pager.host_pages() * PAGE) * u64::from(p.increment)) else {
        return Some(handle);
    };
    let inc = u64::from(p.increment);
    let slot = p.pager.console_slot(((handle - p.gpu_start) / inc) as u32)?;
    Some(p.gpu_start + u64::from(slot) * inc + (handle - p.gpu_start) % inc)
}

/// The CPU address and the increment of the host slot at GPU handle `handle` of a shader visible view
/// heap, where a PC list reads it: in the host pages or the tail of moved tables of a paged heap, or in
/// an unpaged heap and its ring. `None` for any other handle.
pub fn host_slot(handle: u64) -> Option<(usize, u32)> {
    if ANY_PAGED.load(Ordering::Acquire) {
        let g = PAGED.read().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = g.iter().find(|p| handle >= p.gpu_start && handle < p.gpu_start + u64::from(p.host_count) * u64::from(p.increment)) {
            return Some((p.host_cpu + (handle - p.gpu_start) as usize, p.increment));
        }
    }
    let g = VISIBLE.read().unwrap_or_else(|e| e.into_inner());
    let v = g.iter().find(|v| v.heap_type == 0 && handle >= v.gpu_start && handle < v.gpu_start + u64::from(v.host_count + v.ring_slots) * u64::from(v.increment))?;
    Some((v.host_cpu + (handle - v.gpu_start) as usize, v.increment))
}

/// The console slot behind host index `index` of the heap at GPU start `gpu_start` (the index
/// itself for a heap that is not paged).
pub fn console_index(gpu_start: u64, index: u32) -> Option<u32> {
    if !ANY_PAGED.load(Ordering::Acquire) {
        return Some(index);
    }
    let g = PAGED.read().unwrap_or_else(|e| e.into_inner());
    match g.iter().find(|p| p.gpu_start == gpu_start) {
        Some(p) => p.console_slot(index),
        None => Some(index),
    }
}

/// `n` slots from CPU handle `cpu` were written. When `cpu` is in the backing heap of a paged heap,
/// their pages get host pages and the slots go there at the next [`flush`].
pub fn written(cpu: usize, n: u32) {
    written_with_drain(cpu, n, &mut drain_submitted_gpu);
}

fn written_with_drain(cpu: usize, n: u32, drain: &mut dyn FnMut() -> bool) {
    if !ANY_PAGED.load(Ordering::Acquire) {
        return;
    }
    let drain_epochs = {
        let now = PRESENTS.load(Ordering::Relaxed);
        let mut g = PAGED.write().unwrap_or_else(|e| e.into_inner());
        let Some(p) = g.iter_mut().find(|p| cpu >= p.backing_cpu && cpu < p.backing_cpu + p.pager.count as usize * p.increment as usize) else { return };
        let first = ((cpu - p.backing_cpu) / p.increment as usize) as u32;
        if !p.pager.written_needs_drain(first, n, now) {
            if p.pager.written(first, n, now) || p.pager.map_changed() {
                DIRTY.store(true, Ordering::Release);
            }
            return;
        }
        g.iter_mut().map(|ph| (ph.heap, ph.pager.begin_gpu_drain())).collect::<Vec<_>>()
    };
    let idle = drain();
    let now = PRESENTS.load(Ordering::Relaxed);
    let mut g = PAGED.write().unwrap_or_else(|e| e.into_inner());
    if idle {
        for (heap_id, epoch) in drain_epochs {
            if let Some(ph) = g.iter_mut().find(|ph| ph.heap == heap_id) {
                ph.pager.mark_gpu_idle(epoch);
                ph.pager.stats.drained += 1;
            }
        }
    }
    let Some(p) = g.iter_mut().find(|p| cpu >= p.backing_cpu && cpu < p.backing_cpu + p.pager.count as usize * p.increment as usize) else { return };
    let first = ((cpu - p.backing_cpu) / p.increment as usize) as u32;
    if p.pager.written(first, n, now) || p.pager.map_changed() {
        DIRTY.store(true, Ordering::Release);
    }
}

/// Copies the written slots of the backing heaps into the host pages that hold them through `copy`,
/// and writes a changed page map through `map`. Before each draw, dispatch, and queue submission.
pub fn flush(copy: CopyFn, map: MapFn) {
    if !DIRTY.load(Ordering::Acquire) {
        return;
    }
    let mut g = PAGED.write().unwrap_or_else(|e| e.into_inner());
    if !DIRTY.load(Ordering::Relaxed) {
        return;
    }
    for p in g.iter_mut() {
        let fills = p.pager.take_flush_into();
        for &f in &fills {
            run_fill(p, f, copy);
        }
        p.pager.flush_buf = fills;
        if let Some(m) = p.pager.take_map_into() {
            if p.map_cpu != 0 {
                map(p.map_cpu, &m);
            }
            p.pager.map_buf = m;
        }
    }
    DIRTY.store(false, Ordering::Release);
}

/// The extent of a table from its root signature ranges `(count, offset)`: the end of the last
/// range, a range at `None` offset following the one before. `None` when a range is unbounded.
pub fn table_extent(ranges: impl IntoIterator<Item = (u32, Option<u32>)>) -> Option<u32> {
    let mut next = 0u32;
    let mut end = 0u32;
    for (count, offset) in ranges {
        if count == u32::MAX {
            return None;
        }
        let start = offset.unwrap_or(next);
        next = start.checked_add(count)?;
        end = end.max(next);
    }
    Some(end)
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod win {
    use super::{
        ATTR_DESCRIPTOR_HEAP, ATTR_DISPLAY_TARGET, BOUND_HI, BOUND_LO, Entry, TABLE, enabled,
    };
    use crate::log::xs_log;
    use std::ffi::c_void;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicI32, Ordering};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };

    type TitleAllocFn = unsafe extern "system" fn(usize, u64) -> *mut c_void;
    type TitleFreeFn = unsafe extern "system" fn(*mut c_void, u64);

    struct TitleHooks {
        alloc: TitleAllocFn,
        free: TitleFreeFn,
    }

    static HOOKS: OnceLock<Option<TitleHooks>> = OnceLock::new();

    /// The hooks of the title, looked up once. `None` when the switch is off or the title has none
    /// that lie in its own image.
    fn title_hooks() -> Option<&'static TitleHooks> {
        if !enabled() {
            return None;
        }
        HOOKS.get_or_init(adopt).as_ref()
    }

    /// Reads the `[alloc, free]` pair that the main module exports as `__XMemAllocationHooks__`.
    /// The pair is taken only when both functions lie inside the image of that module.
    #[cfg(not(test))]
    fn adopt() -> Option<TitleHooks> {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
        // Offsets in a 64 bit PE image: the NT headers start at the u32 at 0x3C of the DOS header,
        // the signature is "PE\0\0", and SizeOfImage sits at 0x50 inside the NT headers.
        const DOS_MAGIC: u16 = 0x5A4D;
        const PE_SIGNATURE: u32 = 0x0000_4550;
        const NT_HEADERS_POINTER: usize = 0x3C;
        const SIZE_OF_IMAGE: usize = 0x50;
        // SAFETY: reads the header of the main executable image of this process and the two
        // pointer table that it exports.
        unsafe {
            let module = GetModuleHandleW(std::ptr::null());
            if module.is_null() {
                return None;
            }
            let base = module as usize;
            let read_u32 = |at: usize| std::ptr::read_unaligned(at as *const u32);
            if std::ptr::read_unaligned(base as *const u16) != DOS_MAGIC {
                return None;
            }
            let nt = base + read_u32(base + NT_HEADERS_POINTER) as usize;
            if read_u32(nt) != PE_SIGNATURE {
                return None;
            }
            let end = base.saturating_add(read_u32(nt + SIZE_OF_IMAGE) as usize);
            let Some(table) = GetProcAddress(module, c"__XMemAllocationHooks__".as_ptr().cast()) else {
                xs_log!("TITLEHEAP: main module exports no __XMemAllocationHooks__, inert");
                return None;
            };
            let [alloc, free] = std::ptr::read_unaligned(table as usize as *const [usize; 2]);
            let inside = |p: usize| p > base && p < end;
            if !(inside(alloc) && inside(free)) {
                xs_log!(
                    "TITLEHEAP: __XMemAllocationHooks__ entries ({alloc:016X}, {free:016X}) outside main module [{base:016X}, {end:016X}), inert"
                );
                return None;
            }
            xs_log!(
                "TITLEHEAP: adopted __XMemAllocationHooks__ from main module: alloc=module+{:#x} free=module+{:#x}",
                alloc - base,
                free - base
            );
            Some(TitleHooks {
                alloc: std::mem::transmute::<usize, TitleAllocFn>(alloc),
                free: std::mem::transmute::<usize, TitleFreeFn>(free),
            })
        }
    }

    /// The tests of the crate stand in for the title: heap memory comes from `VirtualAlloc`.
    #[cfg(test)]
    fn adopt() -> Option<TitleHooks> {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        unsafe extern "system" fn alloc(bytes: usize, _attributes: u64) -> *mut c_void {
            // SAFETY: a fresh allocation of committed memory.
            unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) }
        }
        unsafe extern "system" fn free(at: *mut c_void, _attributes: u64) {
            // SAFETY: `at` came from `alloc`.
            unsafe { VirtualFree(at, 0, MEM_RELEASE) };
        }
        Some(TitleHooks { alloc, free })
    }

    /// Whether every page of `[base, base + bytes)` is committed and can be read.
    fn range_committed_readable(base: usize, bytes: usize) -> bool {
        let Some(end) = base.checked_add(bytes) else {
            return false;
        };
        let mut at = base;
        while at < end {
            let mut region = MEMORY_BASIC_INFORMATION::default();
            // SAFETY: asks the system about the pages at `at`, nothing is read.
            let got = unsafe { VirtualQuery(at as *const c_void, &mut region, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
            crate::cpu_writes::lazily_committed(&mut region);
            if got == 0 || region.State != MEM_COMMIT || region.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 {
                return false;
            }
            let next = (region.BaseAddress as usize).saturating_add(region.RegionSize);
            if next <= at {
                return false;
            }
            at = next;
        }
        true
    }

    /// Gives the host descriptor heap `host_heap` a range of title memory, taken through the alloc
    /// hook of the title with [`ATTR_DESCRIPTOR_HEAP`], and registers the pair so that handles the
    /// title forms in that range translate to the host heap. `title_count` descriptors are what
    /// the title asked for and `host_count` what the host heap holds (at most as many). Nothing
    /// happens when the title has no hooks or its memory cannot be used.
    pub fn heap_created(host_heap: usize, heap_type: u32, title_count: u32, host_count: u32, inc: u32, host_cpu: usize) {
        static LOGGED: AtomicI32 = AtomicI32::new(0);
        static REFUSED: AtomicI32 = AtomicI32::new(0);
        let Some(hooks) = title_hooks() else {
            return;
        };
        if [title_count, host_count, inc].contains(&0) || host_heap == 0 || host_cpu == 0 {
            return;
        }
        let (Some(title_bytes), Some(host_bytes)) = ((title_count as usize).checked_mul(inc as usize), (host_count as usize).checked_mul(inc as usize)) else {
            return;
        };
        let give_back = |at: *mut c_void| {
            // SAFETY: `at` came from the alloc hook with `ATTR_DESCRIPTOR_HEAP`.
            unsafe { (hooks.free)(at, ATTR_DESCRIPTOR_HEAP) }
        };
        // SAFETY: calls the alloc function of the title.
        let ptr = unsafe { (hooks.alloc)(title_bytes, ATTR_DESCRIPTOR_HEAP) };
        if ptr.is_null() {
            return;
        }
        let base = ptr as usize;
        if !range_committed_readable(base, title_bytes) {
            if REFUSED.fetch_add(1, Ordering::Relaxed) < 16 {
                xs_log!("TITLEHEAP: title memory {base:016X} ({title_bytes:#x} bytes) not committed/readable; keeping host handle");
            }
            give_back(ptr);
            return;
        }
        let entry = Entry { title_base: base, title_bytes, host_cpu, host_bytes, host_heap };
        let registered = {
            let mut table = TABLE.write().unwrap_or_else(|e| e.into_inner());
            let result = table.insert(entry);
            if result.is_ok() {
                BOUND_LO.fetch_min(base, Ordering::Relaxed);
                BOUND_HI.fetch_max(base.saturating_add(title_bytes), Ordering::Relaxed);
            }
            result
        };
        if let Err(why) = registered {
            if REFUSED.fetch_add(1, Ordering::Relaxed) < 16 {
                xs_log!("TITLEHEAP: memory {base:016X} refused: {why}");
            }
            give_back(ptr);
            return;
        }
        crate::mem_owners::add(crate::mem_owners::Kind::TitleDescriptorMemory, crate::mem_owners::Segment::System, 1, title_bytes as u64);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
            xs_log!(
                "TITLEHEAP: heap {host_heap:016X} type={heap_type} count={title_count} (host={host_count}) inc={inc} registered: title [{base:016X}, +{title_bytes:#x}) -> host cpu {host_cpu:016X}"
            );
        }
    }

    /// Unregisters `host_heap` when its proxy is released for good and returns its title memory
    /// through the free hook of the title.
    pub fn heap_released(host_heap: usize) {
        let Some(hooks) = title_hooks() else {
            return;
        };
        let Some(gone) = TABLE.write().unwrap_or_else(|e| e.into_inner()).remove_by_heap(host_heap) else {
            return;
        };
        // SAFETY: `title_base` came from the alloc hook with `ATTR_DESCRIPTOR_HEAP`.
        unsafe { (hooks.free)(gone.title_base as *mut c_void, ATTR_DESCRIPTOR_HEAP) };
        crate::mem_owners::remove(crate::mem_owners::Kind::TitleDescriptorMemory, crate::mem_owners::Segment::System, 1, gone.title_bytes as u64);
    }

    /// Memory for a display target (`ALLOW_DISPLAY` creation on device slot 127), taken through
    /// the alloc hook of the title. 0 when there is none.
    pub fn display_target_alloc(bytes: usize) -> usize {
        match title_hooks() {
            Some(hooks) if bytes != 0 => {
                // SAFETY: calls the alloc function of the title.
                unsafe { (hooks.alloc)(bytes, ATTR_DISPLAY_TARGET) as usize }
            }
            _ => 0,
        }
    }
}

#[cfg(windows)]
pub use win::{display_target_alloc, heap_created, heap_released};

#[cfg(test)]
mod pager_tests {
    use super::*;

    fn fills(p: &mut Pager) -> Vec<Fill> {
        let mut f = p.take_flush();
        f.sort_by_key(|f| (f.to, f.from));
        f
    }

    #[test]
    fn a_ring_larger_than_the_host_heap_is_written_bound_and_recycled_page_by_page() {
        // A ring of 10 pages over 4 host pages and a bit.
        let mut p = Pager::new(10 * PAGE, 4 * PAGE + 7);
        let mut now = 0;
        // The title walks its ring twice, one page per frame: it writes a table, binds it on a
        // list, and resets the list the next frame.
        for round in 0..2 {
            for c in 0..10 {
                let slot = c * PAGE + 5;
                assert!(p.written(slot, 2, now), "round {round} page {c}: the write gets a host page");
                let filled = fills(&mut p);
                assert_eq!(filled.len(), 1);
                assert_eq!((filled[0].from, filled[0].count), (c * PAGE, PAGE), "a newly taken page is filled whole");
                let m = p.map(1, slot, 2, now).expect("the page is held");
                assert!(m.fills.is_empty() && !m.early);
                assert!(m.host + 1 < 4 * PAGE, "inside the host pages");
                assert_eq!(p.console_slot(m.host), Some(slot));
                assert_eq!(filled[0].to, m.host - 5);
                p.release_list(1, now);
                now += KEEP_PRESENTS;
            }
        }
        let s = p.stats();
        assert_eq!((s.highest_written, s.written_pages, s.table_pages, s.index_pages), (Some(9 * PAGE + 6), 10, 10, 0));
        assert_eq!((s.in_use, s.peak, s.host_pages, s.early, s.refused), (4, 4, 4, 0, 0));
        assert_eq!(s.evictions, 16, "20 pages placed in 4 host pages");
    }

    #[test]
    fn pages_read_only_by_index_keep_their_host_page() {
        let mut p = Pager::new(10 * PAGE, 4 * PAGE);
        for c in 0..4 {
            assert!(p.written(c * PAGE, 1, 0));
        }
        assert_eq!(p.take_map().unwrap()[..5], [0, PAGE, 2 * PAGE, 3 * PAGE, UNMAPPED]);
        assert!(!p.written(5 * PAGE, 1, 1000), "no host page is free: the four are kept for good");
        assert!(p.map(1, 6 * PAGE, 1, 1000).is_none(), "not for a table either");
        let s = p.stats();
        assert_eq!((s.index_pages, s.table_pages, s.refused, s.in_use, s.evictions), (5, 0, 2, 4, 0));
    }

    #[test]
    fn tables_at_the_end_of_the_ring_are_clamped_to_it() {
        let mut p = Pager::new(10 * PAGE, 4 * PAGE);
        let last = 10 * PAGE - 1;
        let m = p.map(1, last, 64, 0).unwrap();
        assert_eq!(p.console_slot(m.host), Some(last));
        assert_eq!(m.fills, [Fill { from: 9 * PAGE, to: m.host - (PAGE - 1), count: PAGE }]);
        assert!(p.map(1, 10 * PAGE, 1, 0).is_none(), "past the ring");
        // A ring whose last page is short.
        let mut short = Pager::new(3 * PAGE + 10, 2 * PAGE);
        let m = short.map(1, 3 * PAGE + 9, 4, 0).unwrap();
        assert_eq!(m.fills, [Fill { from: 3 * PAGE, to: m.host - 9, count: 10 }]);
        assert_eq!(short.console_slot(m.host + 1), None, "past the last console slot");
        assert!(!short.written(3 * PAGE + 10, 5, 0), "writes past the ring are left out");
    }

    #[test]
    fn a_table_that_straddles_a_page_boundary_reads_one_contiguous_host_run() {
        let mut p = Pager::new(10 * PAGE, 6 * PAGE);
        // Page 3 goes to host page 0, page 5 (read by index) to host page 1, so page 4 cannot
        // follow page 3 there.
        p.written(3 * PAGE, 1, 0);
        p.written(5 * PAGE, 1, 0);
        fills(&mut p);
        let m = p.map(1, 4 * PAGE - 2, 4, 0).unwrap();
        for k in 0..4 {
            assert_eq!(p.console_slot(m.host + k), Some(4 * PAGE - 2 + k), "descriptor {k}");
        }
        assert_eq!(m.host / PAGE, 2, "pages 3 and 4 together at host pages 2 and 3");
        assert_eq!(m.fills, [Fill { from: 3 * PAGE, to: 2 * PAGE, count: 2 * PAGE }], "one fill for both");
        assert_eq!(p.take_map().unwrap()[3], 0, "the page map keeps naming the first host page of page 3");
        // A write to page 3 reaches both of its host pages, one across the boundary all three.
        assert!(p.written(3 * PAGE + 10, 2, 0));
        assert!(p.written(3 * PAGE + 4, 1, 0));
        assert_eq!(fills(&mut p), [Fill { from: 3 * PAGE + 4, to: 4, count: 8 }, Fill { from: 3 * PAGE + 4, to: 2 * PAGE + 4, count: 8 }]);
        assert!(p.take_flush().is_empty(), "flushed once");
        assert!(p.written(4 * PAGE - 1, 2, 0));
        assert_eq!(fills(&mut p).iter().map(|f| (f.from, f.to, f.count)).collect::<Vec<_>>(), [(4 * PAGE - 1, PAGE - 1, 1), (4 * PAGE - 1, 3 * PAGE - 1, 1), (4 * PAGE, 3 * PAGE, 1)]);
    }

    #[test]
    fn a_run_grows_at_its_end_for_the_next_table_of_a_long_extent() {
        let mut p = Pager::new(10 * PAGE, 4 * PAGE);
        let a = p.map(1, 0, 2 * PAGE, 0).unwrap();
        assert_eq!(a.fills, [Fill { from: 0, to: 0, count: 2 * PAGE }]);
        let b = p.map(1, PAGE, 2 * PAGE, 0).unwrap();
        assert_eq!(b.host, PAGE, "page 1 stays where it is");
        assert_eq!(b.fills, [Fill { from: 2 * PAGE, to: 2 * PAGE, count: PAGE }], "only page 2 is new");
    }

    #[test]
    fn pages_bound_by_a_list_wait_for_its_reset_and_then_for_the_frames_in_flight() {
        let mut p = Pager::new(4 * PAGE, 2 * PAGE);
        p.map(1, 0, 1, 0).unwrap();
        p.map(2, PAGE, 1, 0).unwrap();
        assert!(p.map(3, 2 * PAGE, 1, 100).is_none(), "a list recorded but not yet executed keeps its pages");
        p.release_list(1, 100);
        let early = p.map(3, 2 * PAGE, 1, 101).unwrap();
        assert!(early.early, "taken before the frames in flight are done, as there was no other");
        assert_eq!(early.host, 0);
        p.release_list(2, 200);
        p.release_list(3, 200);
        assert!(p.map(4, 3 * PAGE, 1, 200 + KEEP_PRESENTS - 1).unwrap().early);
        p.release_list(4, 300);
        assert!(!p.map(5, 0, 1, 300 + KEEP_PRESENTS).unwrap().early);
    }

    #[test]
    fn the_page_map_follows_evictions() {
        let mut p = Pager::new(3 * PAGE, PAGE);
        assert_eq!(p.take_map(), Some(vec![UNMAPPED; 3]), "the first map");
        assert_eq!(p.take_map(), None, "unchanged");
        p.map(1, 0, 1, 0).unwrap();
        assert_eq!(p.take_map(), Some(vec![0, UNMAPPED, UNMAPPED]));
        p.release_list(1, 0);
        p.map(1, PAGE + 1, 1, KEEP_PRESENTS).unwrap();
        assert_eq!(p.take_map(), Some(vec![UNMAPPED, 0, UNMAPPED]));
        assert_eq!(p.stats().evictions, 1);
        assert_eq!(identity_map(2 * PAGE + 1), [0, PAGE, 2 * PAGE]);
        assert_eq!(empty_map(PAGE), [UNMAPPED]);
    }

    #[test]
    fn the_extent_of_a_table_follows_its_ranges() {
        assert_eq!(table_extent([(1, None), (2, None)]), Some(3));
        assert_eq!(table_extent([(4, Some(10)), (1, None)]), Some(15));
        assert_eq!(table_extent([(4, Some(10)), (1, Some(0))]), Some(14));
        assert_eq!(table_extent([(1, None), (u32::MAX, None)]), None, "unbounded");
        assert_eq!(table_extent([]), Some(0));
    }

    #[test]
    fn a_paged_heap_gives_host_handles_and_copies_through_the_device() {
        const GPU: u64 = 0x7700_0000_0000;
        const HOST: usize = 0x1000_0000;
        const BACKING: usize = 0x2000_0000;
        let inc = 32u64;
        paged_created(Paged { gpu_start: GPU, increment: 32, host_cpu: HOST, backing_cpu: BACKING, map_cpu: 0x3000, device: 0xD, heap: 0x77, backing: 0x78, backing_charge: None, pager: Pager::new(10 * PAGE, 4 * PAGE), host_count: 4 * PAGE + 100, moved_next: 0, moved: Vec::new() });
        page_map_created(PageMap { heap: 0x77, gpu_start: GPU, address: 0xA000, buffer: 0x79, charge: None });
        assert_eq!(page_map_address(GPU), Some(0xA000));
        let mut copies = Vec::new();
        let mut maps = Vec::new();
        let handle = GPU + u64::from(9 * PAGE + 3) * inc;
        assert!(paged(handle) && !paged(GPU - 32));
        let TableAt::Host(host, false) = table(0x500, handle, 1, &mut |d, dst, src, n| copies.push((d, dst, src, n))) else { panic!("placed") };
        assert!(host < GPU + u64::from(4 * PAGE) * inc, "a handle inside the PC heap");
        let first = ((host - GPU) / inc) as usize - 3;
        assert_eq!(copies, [(0xD, HOST + first * 32, BACKING + (9 * PAGE) as usize * 32, PAGE)]);
        assert_eq!(console_handle(host), Some(handle));
        // a table moved for one draw: its 64 slots from the backing heap into the tail past the pages
        let mut moved = Vec::new();
        let m = move_table(handle, TABLE_WINDOW, &mut |d, dst, src, n| moved.push((d, dst, src, n))).expect("a tail of 100 slots");
        assert_eq!(moved, [(0xD, HOST + (4 * PAGE) as usize * 32, BACKING + (9 * PAGE + 3) as usize * 32, 64)]);
        assert_eq!((m.gpu, m.cpu, m.wrapped), (GPU + u64::from(4 * PAGE) * inc, HOST + (4 * PAGE) as usize * 32, false));
        assert_eq!(console_index(GPU, 4 * PAGE + 5), Some(9 * PAGE + 8), "a slot of the moved table names its console slot");
        assert_eq!(console_index(GPU, 4 * PAGE + 70), None, "past the moved table");
        assert!(move_table(handle, TABLE_WINDOW, &mut |_, _, _, _| {}).is_none(), "36 slots left: the ring would come round over a table younger than KEEP_PRESENTS");
        for _ in 0..=KEEP_PRESENTS {
            present();
        }
        assert_eq!(move_table(handle, TABLE_WINDOW, &mut |_, _, _, _| {}).map(|m| m.wrapped), Some(true), "36 slots left: the ring comes round");
        assert_eq!(console_index(GPU, ((host - GPU) / inc) as u32), Some(9 * PAGE + 3));
        assert_eq!(console_handle(0x1234), Some(0x1234), "outside paged heaps");
        copies.clear();
        written(BACKING + (9 * PAGE + 3) as usize * 32, 1);
        written(BACKING + (2 * PAGE) as usize * 32, 1);
        flush(&mut |d, dst, src, n| copies.push((d, dst, src, n)), &mut |at, m| maps.push((at, m.to_vec())));
        copies.sort();
        assert!(copies.contains(&(0xD, HOST + (first + 3) * 32, BACKING + (9 * PAGE + 3) as usize * 32, 1)), "{copies:x?}");
        assert_eq!(copies.len(), 2, "the written slot and the page taken for the other write");
        assert_eq!(maps.len(), 1);
        assert_eq!((maps[0].0, maps[0].1[9], maps[0].1[0]), (0x3000, first as u32, UNMAPPED));
        list_reset(0x500);
        assert_eq!(paged_released(0x77), Some((0x78, None)));
        assert_eq!(page_map_released(0x77), Some((0x79, None)));
        assert!(!paged(handle));
        assert_eq!(page_map_address(GPU), None);
    }

    #[test]
    fn pager_free_hint_and_scratch_buffers_reuse_capacity() {
        let mut p = Pager::new(6 * PAGE, 3 * PAGE);
        assert!(p.written(0, 1, 0));
        assert!(p.written(2 * PAGE, 1, 0));
        assert_eq!(p.free_hint, 2);
        let fills = p.take_flush_into();
        assert_eq!(fills.len(), 2);
        p.flush_buf = fills;
        let cap = p.flush_buf.capacity();
        assert!(p.written(4 * PAGE, 1, 0));
        assert_eq!(p.free_hint, 0);
        let fills2 = p.take_flush_into();
        assert_eq!(fills2.len(), 1);
        assert_eq!(fills2.capacity(), cap);
        p.flush_buf = fills2;
        let map = p.take_map_into().expect("initial map");
        assert_eq!(map[0], 0);
        assert_eq!(map[2], PAGE);
        assert_eq!(map[4], 2 * PAGE);
        p.map_buf = map;
        assert!(p.take_map_into().is_none());
    }

    #[test]
    fn straddling_table_marks_every_spanned_console_page_as_table_page() {
        let mut p = Pager::new(6 * PAGE, 3 * PAGE);
        // Page 1 is written into host 0 before any table starts in it, then a straddling table
        // spans pages 0..2 into host 1..3.
        assert!(p.written(PAGE + 4, 1, 0));
        let m = p.map(1, PAGE - 2, 4, 0).expect("straddling table spans pages 0 and 1");
        assert!(!m.early);
        p.release_list(1, 0);
        // Because page 1 was spanned by a table, host 0 is no longer pinned forever as index-only,
        // so all three host pages can now be recycled cleanly.
        let next = p.map(2, 2 * PAGE, 3 * PAGE, KEEP_PRESENTS).expect("all three pages can now be recycled");
        assert!(!next.early);
        assert_eq!(p.stats().index_pages, 0);
    }

    #[test]
    fn duplicate_copies_do_not_stay_hot_on_writes_and_evict_before_sole_copies() {
        let mut p = Pager::new(8 * PAGE, 4 * PAGE);
        // Host 0 holds console 0 (canonical), host 1 holds console 5 (index-only, kept).
        assert!(p.written(0, 1, 0));
        assert!(p.written(5 * PAGE, 1, 0));
        // Straddling table on console pages 0..2 allocates a contiguous duplicate at host 2..4.
        let m = p.map(1, PAGE - 2, 4, 0).unwrap();
        assert_eq!(m.host / PAGE, 2);
        p.release_list(1, 0);
        // Repeated writes to console page 0 keep canonical host 0 hot, NOT duplicate host 2.
        for t in 1..=KEEP_PRESENTS {
            assert!(p.written(10, 1, t));
        }
        // Console page 1 (sole copy at host 3) is also cold at KEEP_PRESENTS, while host 2 is a
        // redundant duplicate of console page 0: host 2 must be evicted before host 3 so page 1
        // stays mapped in the page map.
        let placed = p.map(2, 4 * PAGE, 1, KEEP_PRESENTS).expect("reclaims redundant duplicate");
        assert!(!placed.early);
        assert_eq!(placed.host / PAGE, 2, "redundant duplicate evicted before sole copy");
        let map = p.take_map().unwrap();
        assert_eq!(map[0], 0, "canonical copy of console 0 stays intact");
        assert_eq!(map[1], 3 * PAGE, "sole copy of console 1 stays mapped");
        assert_eq!(map[4], 2 * PAGE);
    }

    #[test]
    fn table_and_written_drain_gpu_outside_lock_before_early_eviction() {
        const GPU: u64 = 0x8800_0000_0000;
        const HOST: usize = 0x3100_0000;
        const BACKING: usize = 0x3200_0000;
        const HEAP: usize = 0x8801;
        let inc = 32u64;
        paged_created(Paged {
            gpu_start: GPU,
            increment: 32,
            host_cpu: HOST,
            backing_cpu: BACKING,
            map_cpu: 0,
            device: 0xD,
            heap: HEAP,
            backing: 0x8802,
            backing_charge: None,
            pager: Pager::new(6 * PAGE, 2 * PAGE),
            host_count: 2 * PAGE,
            moved_next: 0,
            moved: Vec::new(),
        });
        // Fill both host pages and immediately unpin them within the same present (< KEEP_PRESENTS).
        assert!(matches!(table_with_drain(1, GPU, 1, &mut |_, _, _, _| {}, &mut || panic!("no drain")), TableAt::Host(_, false)));
        assert!(matches!(table_with_drain(2, GPU + u64::from(PAGE) * inc, 1, &mut |_, _, _, _| {}, &mut || panic!("no drain")), TableAt::Host(_, false)));
        list_reset(1);
        list_reset(2);

        // Next table needs an eviction before KEEP_PRESENTS: must drop PAGED lock and drain once.
        let mut drains = 0u32;
        let res = table_with_drain(3, GPU + u64::from(2 * PAGE) * inc, 1, &mut |_, _, _, _| {}, &mut || {
            drains += 1;
            // Proves PAGED lock is NOT held during the GPU drain callback.
            assert!(PAGED.try_write().is_ok(), "PAGED write lock must be dropped while draining GPU");
            true
        });
        assert_eq!(drains, 1);
        assert!(matches!(res, TableAt::Host(_, false)), "drained page is cold, not early");

        // Second unpinned page from before the drain is already cold: written() reuses it with 0 drains.
        written_with_drain(BACKING + (3 * PAGE) as usize * 32, 1, &mut || {
            drains += 1;
            true
        });
        assert_eq!(drains, 1, "both pre-drain unpinned pages became cold from one drain");

        // Unpin list 3 and test fallback when drain fails (returns false): must fall back to early.
        list_reset(3);
        let fallback = table_with_drain(4, GPU + u64::from(4 * PAGE) * inc, 1, &mut |_, _, _, _| {}, &mut || {
            drains += 1;
            false
        });
        assert_eq!(drains, 2);
        assert!(matches!(fallback, TableAt::Host(_, true)), "failed drain falls back to early eviction");
        list_reset(4);
        let stats = PAGED.read().unwrap().iter().find(|p| p.heap == HEAP).unwrap().pager.stats();
        assert_eq!(stats.drained, 1);
        assert_eq!(stats.early, 1);
        assert_eq!(paged_released(HEAP), Some((0x8802, None)));
    }

    #[test]
    fn flush_waits_for_an_in_flight_flush_to_finish_before_returning() {
        use std::sync::{Arc, Barrier};
        use std::sync::atomic::{AtomicBool, Ordering};
        const GPU: u64 = 0x9900_0000_0000;
        const HOST: usize = 0x4100_0000;
        const BACKING: usize = 0x4200_0000;
        const HEAP: usize = 0x9901;
        paged_created(Paged {
            gpu_start: GPU,
            increment: 32,
            host_cpu: HOST,
            backing_cpu: BACKING,
            map_cpu: 0,
            device: 0xD,
            heap: HEAP,
            backing: 0x9902,
            backing_charge: None,
            pager: Pager::new(2 * PAGE, 2 * PAGE),
            host_count: 2 * PAGE,
            moved_next: 0,
            moved: Vec::new(),
        });
        written_with_drain(BACKING, 1, &mut || true);

        let entered_copy = Arc::new(Barrier::new(2));
        let copy_done = Arc::new(AtomicBool::new(false));
        let entered_clone = Arc::clone(&entered_copy);
        let done_clone = Arc::clone(&copy_done);

        let t1 = std::thread::spawn(move || {
            flush(
                &mut |_, _, _, _| {
                    entered_clone.wait();
                    std::thread::sleep(std::time::Duration::from_millis(25));
                    done_clone.store(true, Ordering::Release);
                },
                &mut |_, _| {},
            );
        });

        // Wait until t1 is mid-copy inside flush(), then invoke flush() on this thread.
        entered_copy.wait();
        flush(&mut |_, _, _, _| {}, &mut |_, _| {});
        assert!(copy_done.load(Ordering::Acquire), "concurrent flush must not return before in-flight copy finishes");
        t1.join().unwrap();
        assert_eq!(paged_released(HEAP), Some((0x9902, None)));
    }
}

#[cfg(test)]
mod tests {
    use super::{Entry, Table};

    #[test]
    fn the_ring_of_moved_tables_leaves_the_pager_its_peak_pages() {
        // A view heap of 1,200,000 slots on Windows is clamped to 1,000,000 slots, and the menus
        // use a few hundred host pages at their peak.
        let pager = super::Pager::new(1_200_000, 1_000_000 - super::MOVED_SLOTS);
        assert!(pager.host_pages() >= 200, "{} host pages", pager.host_pages());
        let tail = 1_000_000 - pager.host_pages() * super::PAGE;
        assert!(tail >= super::MOVED_SLOTS && tail / 64 >= 256, "a tail of {tail} slots");
    }

    #[test]
    fn a_moved_table_is_never_overwritten_before_keep_presents_have_passed() {
        let moved = [(100, 0, 64, 10u64), (164, 64, 64, 12)];
        assert!(super::overwrites_young(&moved, 100, 64, 10 + super::KEEP_PRESENTS - 1), "written 5 presents ago");
        assert!(!super::overwrites_young(&moved, 100, 64, 10 + super::KEEP_PRESENTS), "6 presents ago: free");
        assert!(super::overwrites_young(&moved, 150, 30, 10 + super::KEEP_PRESENTS), "the second one overlaps and is younger");
        assert!(!super::overwrites_young(&moved, 228, 64, 11), "a run that touches none");
        assert!(!super::overwrites_young(&[], 0, 64, 0));
    }

    #[test]
    fn moved_tables_take_runs_of_the_tail_in_turn() {
        // a tail of 200 slots from 1000
        assert_eq!(super::moved_run(1000, 1200, 0, 64), Some((1000, false)));
        assert_eq!(super::moved_run(1000, 1200, 1128, 64), Some((1128, false)));
        assert_eq!(super::moved_run(1000, 1200, 1192, 64), Some((1000, true)), "past the end it comes round");
        assert_eq!(super::moved_run(1000, 1040, 0, 64), None, "no room");
        assert_eq!(super::moved_run(1000, 1200, 0, 0), None);
    }

    // Each test below registers its own heap at a distinct `gpu_start`/`host_cpu`/`device` (the
    // registry is a global shared with every other test in this binary, which may run in
    // parallel) so the entries never overlap and move_table's range search cannot cross them.
    fn visible_fixture(gpu_start: u64, host_cpu: usize, device: usize, ring_slots: u32) -> super::Visible {
        super::Visible { gpu_start, increment: 32, host_cpu, host_count: 1000, title_base: 0x5000_0000, title_count: 1000, heap_type: 0, device, heap: gpu_start as usize, ring_slots, ring_next: 0 }
    }

    #[test]
    fn a_host_slot_of_an_unpaged_heap_is_found_in_its_slots_and_its_ring() {
        let v = visible_fixture(0x4700_0000, 0x9700_0000, 0x777, 640);
        super::visible_created(v);
        assert_eq!(super::host_slot(v.gpu_start + 5 * 32), Some((v.host_cpu + 5 * 32, 32)));
        assert_eq!(super::host_slot(v.gpu_start + 1200 * 32), Some((v.host_cpu + 1200 * 32, 32)), "a slot of the ring");
        assert_eq!(super::host_slot(v.gpu_start + 1640 * 32), None, "past the ring");
        assert_eq!(super::host_slot(v.gpu_start - 32), None, "before the heap");
    }

    #[test]
    fn move_table_finds_the_ring_of_an_unpaged_heap_and_copies_from_its_own_slots() {
        let v = visible_fixture(0x4100_0000, 0x9100_0000, 0x111, super::MOVED_SLOTS);
        super::visible_created(v);
        let console = v.gpu_start + 7 * 32;
        let mut copied = Vec::new();
        let m = super::move_table(console, 64, &mut |d, dst, src, n| copied.push((d, dst, src, n))).expect("a ring past host_count");
        assert_eq!(copied, [(0x111, v.host_cpu + 1000 * 32, v.host_cpu + 7 * 32, 64)], "copied from the heap's own host slots, no backing heap");
        assert_eq!((m.gpu, m.cpu, m.wrapped), (v.gpu_start + 1000 * 32, v.host_cpu + 1000 * 32, false));
        // title slot count and host_count are unaffected by the ring
        assert_eq!((v.title_count, v.host_count), (1000, 1000));
    }

    #[test]
    fn move_table_wraps_the_ring_of_an_unpaged_heap_when_it_runs_out() {
        let v = visible_fixture(0x4200_0000, 0x9200_0000, 0x222, 640);
        super::visible_created(v);
        let console = v.gpu_start + 3 * 32;
        // 640 slots of 64 fit 10 times exactly; the 11th comes round, once KEEP_PRESENTS presents
        // have passed over the tables written (before that the ring is full and refuses).
        for i in 0..10 {
            let m = super::move_table(console, 64, &mut |_, _, _, _| {}).unwrap_or_else(|| panic!("round {i}"));
            assert!(!m.wrapped, "round {i}: still room in the ring");
        }
        for _ in 0..=super::KEEP_PRESENTS {
            super::present();
        }
        let m = super::move_table(console, 64, &mut |_, _, _, _| {}).expect("the ring still has room, just come round");
        assert!(m.wrapped, "the ring came round to its start");
        assert_eq!(m.cpu, v.host_cpu + 1000 * 32, "back at the start of the tail");
    }

    #[test]
    fn move_table_finds_no_ring_when_the_extra_slots_were_refused() {
        // The heap got no MOVED_SLOTS extra tail (PC refused the bigger allocation): no ring.
        let v = visible_fixture(0x4300_0000, 0x9300_0000, 0x333, 0);
        super::visible_created(v);
        let console = v.gpu_start + 7 * 32;
        assert_eq!(super::move_table(console, 64, &mut |_, _, _, _| {}), None);
        // the heap is still registered at its title count, unaffected
        assert_eq!((v.title_count, v.host_count, v.ring_slots), (1000, 1000, 0));
    }

    #[test]
    fn unaligned_constant_buffer_views_are_noted_per_host_slot() {
        super::note_cbv(0xAB00, Some((0x1_0020, 64)));
        assert_eq!(super::unaligned_cbv(0xAB00), Some((0x1_0020, 64)));
        super::note_cbv(0xAB00, None);
        assert_eq!(super::unaligned_cbv(0xAB00), None);
    }

    #[test]
    fn a_tag_names_its_host_slot_and_other_bytes_are_no_tag() {
        assert_eq!(super::tagged_word(super::tag_words(0x1234_5000)), Some(0x1234_5000));
        assert_eq!(super::tagged_word([0, 0, 0]), None);
        let mut w = super::tag_words(0x1234_5000);
        w[2] ^= 1;
        assert_eq!(super::tagged_word(w), None);
    }

    #[test]
    fn a_table_copies_the_tagged_slots_of_its_window_into_the_host_heap() {
        let v = super::Visible { gpu_start: 0x1_0000, increment: 32, host_cpu: 0x9000_0000, host_count: 100, title_base: 0x5000_0000, title_count: 120, heap_type: 0, device: 0, heap: 0, ring_slots: 0, ring_next: 0 };
        let read = |a: usize| match (a - 0x5000_0000) / 32 {
            10 => super::tag_words(0x7000_0000),
            12 => super::tag_words(0x9000_0000 + 12 * 32),
            13 => [1, 2, 3],
            _ => [0; 3],
        };
        assert_eq!(super::table_copies(&v, 0x1_0000 + 10 * 32, read), [super::TableSlot { dst: 0x9000_0000 + 10 * 32, word: 0x7000_0000 }, super::TableSlot { dst: 0x9000_0000 + 12 * 32, word: 0x9000_0000 + 12 * 32 }]);
        assert_eq!(super::table_copies(&v, 0x1_0000 + 99 * 32, |_| super::tag_words(0x7000_0000)).len(), 1, "slots past the host heap are left out");
        assert!(super::table_copies(&v, 0x100, read).is_empty());
    }

    #[test]
    fn handles_in_a_registered_heap_translate_by_offset() {
        let mut t = Table::default();
        let e = Entry { title_base: 0x7000_0000, title_bytes: 4096 * 32, host_cpu: 0x2000_0000, host_bytes: 1000 * 32, host_heap: 0xBEEF };
        assert_eq!(t.insert(e), Ok(()));
        assert_eq!(t.translate(0x7000_0000), Some(Some(0x2000_0000)));
        assert_eq!(t.translate(0x7000_0000 + 96), Some(Some(0x2000_0060)));
        assert_eq!(t.translate(0x7000_0000 + 999 * 32), Some(Some(0x2000_0000 + 999 * 32)));
        assert_eq!(t.translate(0x7000_0000 + 1000 * 32), Some(None), "past the host heap but inside the title heap");
        assert_eq!(t.translate(0x7000_0000 + 4096 * 32), None, "the first byte after the title heap");
        assert_eq!(t.translate(0x6FFF_FFFF), None);
        assert_eq!(t.translate(0), None);
    }

    #[test]
    fn overlapping_or_empty_ranges_are_refused() {
        let mut t = Table::default();
        let e = Entry { title_base: 0x1000, title_bytes: 0x1000, host_cpu: 0x9000, host_bytes: 0x1000, host_heap: 1 };
        assert_eq!(t.insert(e), Ok(()));
        assert!(t.insert(Entry { title_base: 0x1800, host_heap: 2, ..e }).is_err(), "starts inside");
        assert!(t.insert(Entry { title_base: 0x800, host_heap: 3, ..e }).is_err(), "ends inside");
        assert!(t.insert(Entry { title_base: 0x1000, host_heap: 4, ..e }).is_err(), "same range");
        assert!(t.insert(Entry { title_bytes: 0, title_base: 0x5000, host_heap: 5, ..e }).is_err());
        assert!(t.insert(Entry { host_cpu: 0, title_base: 0x5000, host_heap: 6, ..e }).is_err());
        assert!(t.insert(Entry { title_base: usize::MAX - 8, host_heap: 7, ..e }).is_err(), "wraps around");
        assert_eq!(t.insert(Entry { title_base: 0x2000, host_heap: 8, ..e }), Ok(()), "adjacent ranges are fine");
        assert_eq!(t.insert(Entry { title_base: 0, title_bytes: 0x1000, host_heap: 9, ..e }), Ok(()));
    }

    #[test]
    fn a_heap_is_found_and_removed_by_its_host_handle() {
        let mut t = Table::default();
        let a = Entry { title_base: 0x4000, title_bytes: 0x100, host_cpu: 0x10, host_bytes: 0x100, host_heap: 0xA };
        let b = Entry { title_base: 0x2000, host_heap: 0xB, ..a };
        t.insert(a).unwrap();
        t.insert(b).unwrap();
        assert_eq!(t.title_start_for_heap(0xA), Some(0x4000));
        assert_eq!(t.title_start_for_heap(0xC), None);
        assert_eq!(t.remove_by_heap(0xA), Some(a));
        assert_eq!(t.remove_by_heap(0xA), None);
        assert_eq!(t.translate(0x4000), None);
        assert_eq!(t.translate(0x2000), Some(Some(0x10)), "the other heap stays");
    }

    #[test]
    fn pager_release_list_reuses_pinned_list_capacity() {
        let mut p = super::Pager::new(4 * super::PAGE, 4 * super::PAGE);
        let list = 0xDEAD_0001usize;
        assert!(p.map(list, 0, super::PAGE * 2, 0).is_some());
        let cap = p.lists.get(&list).map(Vec::capacity).unwrap();
        assert!(cap >= 2);
        p.release_list(list, 1);
        let after = p.lists.get(&list).map(|v| (v.len(), v.capacity()));
        assert_eq!(after, Some((0, cap)));
    }
}
