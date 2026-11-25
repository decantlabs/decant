//! The flicker probe behind `DECANT_D3D12X_FLICKER_PROBE`. It tells apart two causes of a picture that
//! changes from one frame to the next: a
//! descriptor that is rewritten before the GPU has read it, and title memory that a sync copies into
//! the GPU mirror while a list that reads it still runs.
//!
//! For every graphics draw the probe notes the host descriptor slots of the bound resource tables (the
//! first slots of each table, up to its extent) with a fingerprint of their bytes. It also notes the
//! title ranges the draw reads through the mirror: the vertices and indices of the bound buffers and the
//! aligned constant buffer views of the tables. At each submission it signals a fence of its own on the
//! queue and keeps the reads until that fence is reached. It counts four things:
//!
//! * `in list`: a slot that an earlier draw of the same list reads gets another descriptor before the
//!   list runs. The PC GPU reads a slot when the draw runs, so both draws see the later descriptor.
//! * `before submit`: a slot changes between the last draw that reads it and the submission.
//! * `slot in flight`: a slot changes while the list that reads it runs on the GPU.
//! * `mirror in flight`: title bytes that a running list reads change, and the sync of a later
//!   submission copies them into the mirror before that list is done.
//!
//! A slot is judged by fingerprint first and by the fence after, so a change seen while the fence is
//! not reached did happen while the list ran.
//!
//! `DECANT_D3D12X_FLICKER_PROBE=wait` also makes the CPU wait for each submission to finish (a known
//! technique). Then no list runs while the CPU writes. A flicker that stays has
//! another cause: presentation, or bindings that are already wrong when recorded. A submission that
//! does not finish within 2 s lets the CPU go on, and after 4 of them the waits stop.
//!
//! The memory reads need `DECANT_D3D12X_TITLE_MEM=1` (no mirror without it). The table slots need the user
//! data mirror (`DECANT_D3D12X_DXIL_OVERRIDE`). Targets and pipelines in the lines need
//! `DECANT_D3D12X_PRESENT_PROBE=1`, they read 0 without it. The probe turns on the view records of
//! `view_diag` and the buffer records of `draw_diag`, which name the resources in the lines.
//!
//! Off by default. Off, each hook is one relaxed atomic load.
#![allow(unsafe_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::{D3D12_FENCE_FLAG_NONE, ID3D12CommandQueue, ID3D12Device, ID3D12Fence};
use windows::core::Interface;

use crate::log::xs_log;

/// Slots looked at per table. The draws of a retail title read offsets 0 to 3.
const TABLE_SLOTS: u32 = 8;
/// Slot reads kept per list, and draws whose target is kept.
const MAX_SLOTS: usize = 8192;
/// Memory reads kept per list.
const MAX_MEMS: usize = 4096;
/// The longest title range read for one fingerprint.
const MAX_RANGE: u32 = 256 << 10;
/// Submissions kept while their fence is not reached.
const MAX_BATCHES: usize = 256;
/// The most indices read to find the vertices an indexed draw reads.
const MAX_INDICES: u32 = 1 << 16;
/// Presents between two count lines.
const REPORT_PRESENTS: u64 = 128;
/// The longest CPU wait for one submission in the wait mode.
const WAIT_LIMIT: Duration = Duration::from_secs(2);
/// Late submissions after which the wait mode stops waiting.
const LATE_LIMIT: u64 = 4;

const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const WATCH: u8 = 2;
const WAIT: u8 = 3;

static MODE: AtomicU8 = AtomicU8::new(UNKNOWN);

/// The mode a value of `DECANT_D3D12X_FLICKER_PROBE` asks for: off when unset, empty or 0, the CPU wait
/// for `wait`, the watch for anything else.
fn parse_mode(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        None | Some("" | "0") => OFF,
        Some(v) if v.eq_ignore_ascii_case("wait") => WAIT,
        Some(_) => WATCH,
    }
}

fn mode() -> u8 {
    match MODE.load(Ordering::Relaxed) {
        UNKNOWN => {
            let m = parse_mode(std::env::var("DECANT_D3D12X_FLICKER_PROBE").ok().as_deref());
            MODE.store(m, Ordering::Relaxed);
            if m != OFF {
                let what = if m == WAIT { "wait (the CPU waits for each submission)" } else { "watch" };
                xs_log!("flicker probe: on, mode {what}, user data mirror {}, title memory {}", crate::user_mirror::active(), crate::title_mem::enabled());
            }
            m
        }
        m => m,
    }
}

/// Whether the probe is on.
pub fn on() -> bool {
    mode() >= WATCH
}

/// A fingerprint of `bytes` (FNV 1a over 64 bit words, then the bytes left).
fn fingerprint(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let (words, rest) = bytes.as_chunks::<8>();
    for w in words {
        h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(PRIME);
    }
    for &b in rest {
        h = (h ^ u64::from(b)).wrapping_mul(PRIME);
    }
    (h ^ bytes.len() as u64).wrapping_mul(PRIME)
}

/// A descriptor slot a draw reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotRead {
    /// The host slot (its CPU address in the shader visible heap the PC list reads) and its bytes.
    pub host: usize,
    pub len: u32,
    /// The slot that `view_diag` keys the view by (the title's slot in the backing or host heap), 0
    /// when not known, and the PC resource of that view when the draw was recorded.
    pub console: usize,
    pub res: usize,
    /// The fingerprint of the host slot's bytes.
    pub print: u64,
    /// The draw of its list (from 1), the root parameter and the offset in the table.
    pub draw: u32,
    pub param: u16,
    pub offset: u16,
}

/// What a memory read is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Vertex(u32),
    Index,
    Cbv,
}

/// A title range a draw reads through the mirror.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemRead {
    pub va: u64,
    pub len: u32,
    pub draw: u32,
    pub source: Source,
}

/// The render target and the pipeline of a draw, zeros when not known.
type Target = (usize, usize);

/// The reads of one list since its reset.
#[derive(Default)]
struct ListReads {
    draws: u32,
    targets: Vec<Target>,
    slots: Vec<SlotRead>,
    by_host: HashMap<usize, usize>,
    mems: Vec<MemRead>,
    ranges: HashSet<(u64, u32)>,
}

fn target_of(targets: &[Target], draw: u32) -> Target {
    draw.checked_sub(1).and_then(|i| targets.get(i as usize)).copied().unwrap_or_default()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeldSlot {
    list: usize,
    target: Target,
    read: SlotRead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeldMem {
    list: usize,
    target: Target,
    read: MemRead,
    /// The fingerprint of the title bytes at the submission, which the mirror then held.
    print: u64,
}

/// The reads of one submission, kept until its fence is reached.
#[derive(Debug)]
pub struct Batch {
    queue: usize,
    fence: u64,
    present: u64,
    slots: Vec<HeldSlot>,
    mems: Vec<HeldMem>,
}

/// Where a submission was when a change was seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flight {
    pub fence: u64,
    pub done: u64,
    pub presents: u64,
}

/// One change the probe saw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finding {
    InList { list: usize, first: SlotRead, later: SlotRead, targets: [Target; 2] },
    BeforeSubmit { list: usize, read: SlotRead, now: u64, target: Target },
    SlotInFlight { list: usize, read: SlotRead, now: u64, flight: Flight, target: Target },
    MirrorInFlight { list: usize, read: MemRead, was: u64, now: u64, flight: Flight, target: Target },
}

/// Title bytes of a running submission that changed before a sync: the sync copies them into the
/// mirror. Confirmed after the sync when the fence is still not reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Suspect {
    queue: usize,
    fence: u64,
    index: usize,
    now: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub draws: u64,
    pub slot_reads: u64,
    pub mem_reads: u64,
    pub in_list: u64,
    pub before_submit: u64,
    pub slot_in_flight: u64,
    pub mirror_in_flight: u64,
    pub batches: u64,
    pub retired: u64,
    pub most_in_flight: u64,
    pub dropped: u64,
}

/// The bookkeeping, apart from any memory or fence access (tested on synthetic data).
#[derive(Default)]
pub struct Core {
    lists: HashMap<usize, ListReads>,
    flight: VecDeque<Batch>,
    counts: Counts,
}

impl Core {
    /// `list` records a draw of `target` that reads `slots` and `mems` (their `draw` is set here).
    pub fn draw(&mut self, list: usize, target: Target, slots: &[SlotRead], mems: &[MemRead]) -> Vec<Finding> {
        let c = &mut self.counts;
        c.draws += 1;
        let l = self.lists.entry(list).or_default();
        l.draws += 1;
        let draw = l.draws;
        if l.targets.len() < MAX_SLOTS {
            l.targets.push(target);
        }
        let mut out = Vec::new();
        for s in slots {
            let read = SlotRead { draw, ..*s };
            match l.by_host.get(&s.host) {
                Some(&i) => {
                    let first = l.slots[i];
                    if first.print != read.print {
                        c.in_list += 1;
                        out.push(Finding::InList { list, first, later: read, targets: [target_of(&l.targets, first.draw), target] });
                        l.slots[i] = read;
                    }
                }
                None if l.slots.len() < MAX_SLOTS => {
                    l.by_host.insert(s.host, l.slots.len());
                    l.slots.push(read);
                    c.slot_reads += 1;
                }
                None => c.dropped += 1,
            }
        }
        for m in mems {
            if l.ranges.contains(&(m.va, m.len)) {
                continue;
            }
            if l.mems.len() >= MAX_MEMS {
                c.dropped += 1;
                continue;
            }
            l.ranges.insert((m.va, m.len));
            l.mems.push(MemRead { draw, ..*m });
            c.mem_reads += 1;
        }
        out
    }

    /// `list` was reset or released. Submissions in flight keep their own copy of its reads.
    pub fn forget(&mut self, list: usize) {
        self.lists.remove(&list);
    }

    /// The title ranges of running submissions whose bytes `print` finds changed since their submission.
    pub fn suspects(&self, print: impl Fn(u64, u32) -> Option<u64>) -> Vec<Suspect> {
        let mut out = Vec::new();
        for b in &self.flight {
            for (index, h) in b.mems.iter().enumerate() {
                if let Some(now) = print(h.read.va, h.read.len)
                    && now != h.print
                {
                    out.push(Suspect { queue: b.queue, fence: b.fence, index, now });
                }
            }
        }
        out
    }

    /// The suspects whose submission has still not reached its fence after the sync (`done` gives the
    /// completed value of a queue's fence): the sync copied their bytes while the lists ran.
    pub fn confirm(&mut self, suspects: &[Suspect], present: u64, done: impl Fn(usize) -> u64) -> Vec<Finding> {
        let mut out = Vec::new();
        for s in suspects {
            let Some(b) = self.flight.iter_mut().find(|b| b.queue == s.queue && b.fence == s.fence) else { continue };
            let d = done(b.queue);
            if d >= b.fence {
                continue;
            }
            let flight = Flight { fence: b.fence, done: d, presents: present.saturating_sub(b.present) };
            let Some(h) = b.mems.get_mut(s.index) else { continue };
            if h.print == s.now {
                continue;
            }
            self.counts.mirror_in_flight += 1;
            out.push(Finding::MirrorInFlight { list: h.list, read: h.read, was: h.print, now: s.now, flight, target: h.target });
            h.print = s.now;
        }
        out
    }

    /// Compares the slots of running submissions with `print`, then reads the fences through `done`.
    /// A submission whose fence is reached is dropped.
    pub fn check(&mut self, present: u64, print: impl Fn(usize, u32) -> Option<u64>, done: impl Fn(usize) -> u64) -> Vec<Finding> {
        let now: Vec<Vec<Option<u64>>> = self.flight.iter().map(|b| b.slots.iter().map(|h| print(h.read.host, h.read.len)).collect()).collect();
        let mut out = Vec::new();
        let mut keep = VecDeque::with_capacity(self.flight.len());
        for (mut b, now) in self.flight.drain(..).zip(now) {
            let d = done(b.queue);
            if d >= b.fence {
                self.counts.retired += 1;
                continue;
            }
            let flight = Flight { fence: b.fence, done: d, presents: present.saturating_sub(b.present) };
            for (h, n) in b.slots.iter_mut().zip(now) {
                if let Some(n) = n
                    && n != h.read.print
                {
                    self.counts.slot_in_flight += 1;
                    out.push(Finding::SlotInFlight { list: h.list, read: h.read, now: n, flight, target: h.target });
                    h.read.print = n;
                }
            }
            keep.push_back(b);
        }
        self.flight = keep;
        out
    }

    /// `lists` are about to run on `queue`: their slots are compared with `slot_print` and the title
    /// bytes of their memory reads taken with `mem_print`. The reads to keep until the fence, if any.
    pub fn submit(
        &mut self,
        queue: usize,
        lists: &[usize],
        present: u64,
        slot_print: impl Fn(usize, u32) -> Option<u64>,
        mem_print: impl Fn(u64, u32) -> Option<u64>,
    ) -> (Vec<Finding>, Option<Batch>) {
        let mut out = Vec::new();
        let mut batch = Batch { queue, fence: 0, present, slots: Vec::new(), mems: Vec::new() };
        for &list in lists {
            let Some(l) = self.lists.get_mut(&list) else { continue };
            for s in &mut l.slots {
                let target = target_of(&l.targets, s.draw);
                if let Some(now) = slot_print(s.host, s.len)
                    && now != s.print
                {
                    self.counts.before_submit += 1;
                    out.push(Finding::BeforeSubmit { list, read: *s, now, target });
                    s.print = now;
                }
                batch.slots.push(HeldSlot { list, target, read: *s });
            }
            for m in &l.mems {
                if let Some(print) = mem_print(m.va, m.len) {
                    batch.mems.push(HeldMem { list, target: target_of(&l.targets, m.draw), read: *m, print });
                }
            }
        }
        let batch = (!batch.slots.is_empty() || !batch.mems.is_empty()).then_some(batch);
        (out, batch)
    }

    /// `batch` was submitted and its queue signals `fence` once it is done.
    pub fn push(&mut self, mut batch: Batch, fence: u64) {
        batch.fence = fence;
        self.counts.batches += 1;
        self.flight.push_back(batch);
        while self.flight.len() > MAX_BATCHES {
            self.flight.pop_front();
            self.counts.dropped += 1;
        }
        self.counts.most_in_flight = self.counts.most_in_flight.max(self.flight.len() as u64);
    }
}

/// The lowest and highest index of `bytes`, indices of `stride` bytes (2 or 4).
fn index_span(bytes: &[u8], stride: usize) -> Option<(u32, u32)> {
    let values = bytes.chunks_exact(stride).map(|c| if stride == 4 { u32::from_le_bytes([c[0], c[1], c[2], c[3]]) } else { u32::from(u16::from_le_bytes([c[0], c[1]])) });
    values.fold(None, |acc, v| match acc {
        None => Some((v, v)),
        Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
    })
}

/// The part of a buffer at `va` of `size` bytes that vertices `lo..=hi` of `stride` bytes take (the
/// whole buffer when they are not known), at most [`MAX_RANGE`] bytes. `None` when nothing is left.
fn vertex_range(va: u64, size: u64, stride: u64, vertices: Option<(i64, i64)>) -> Option<(u64, u32)> {
    if va == 0 || size == 0 {
        return None;
    }
    let (start, end) = match vertices {
        Some((lo, hi)) if stride > 0 && lo <= hi => {
            let start = u64::try_from(lo.max(0)).ok()?.saturating_mul(stride);
            let end = u64::try_from(hi.saturating_add(1).max(0)).ok()?.saturating_mul(stride);
            (start.min(size), end.min(size))
        }
        Some(_) => return None,
        None => (0, size),
    };
    let len = end.checked_sub(start).filter(|&n| n > 0)?;
    Some((va + start, len.min(u64::from(MAX_RANGE)) as u32))
}

/// The arguments of a graphics draw, for the vertices and indices it reads.
#[derive(Clone, Copy, Debug)]
pub enum Draw {
    Plain { vertices: u32, first: u32 },
    Indexed { count: u32, first: u32, base: i32 },
    /// `DrawIndexedX`: `count` indices of `format` at title address `va`.
    IndexedAt { va: u64, format: u32, count: u32 },
}

static CORE: Mutex<Option<Core>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Core) -> R) -> R {
    let mut g = CORE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(Core::default))
}

/// Safely copies `len` bytes from `at` in the current process into an owned buffer.
fn read_bytes(at: usize, len: usize) -> Option<Vec<u8>> {
    if at == 0 || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    let mut read = 0usize;
    // SAFETY: `buf` is a valid writable buffer of `len` bytes; `ReadProcessMemory` on the
    // current process copies without creating a shared Rust slice over concurrent memory.
    let ok = unsafe {
        windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            at as *const c_void,
            buf.as_mut_ptr().cast(),
            len,
            &mut read,
        )
    };
    (ok != 0 && read == len).then_some(buf)
}

/// The fingerprint of `len` bytes at `at`, `None` when they are not all readable.
fn print_at(at: usize, len: u32) -> Option<u64> {
    read_bytes(at, len as usize).map(|bytes| fingerprint(&bytes))
}

fn print_title(va: u64, len: u32) -> Option<u64> {
    print_at(usize::try_from(va).ok()?, len)
}

/// The fence the probe signals on one PC queue and the last value signalled.
struct QueueFence {
    fence: ID3D12Fence,
    value: u64,
}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);

/// The completed value of the probe's fence of `queue`. A queue without one counts as done, so that
/// nothing is called in flight without a fence to show it.
fn done(queue: usize) -> u64 {
    let g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live fence the probe made.
    g.as_ref().and_then(|m| m.get(&queue)).map_or(u64::MAX, |q| unsafe { q.fence.GetCompletedValue() })
}

/// Signals the next value of the probe's fence on `queue` (made on first use): the value.
fn signal(queue: *mut c_void) -> Option<u64> {
    static FAILED: AtomicU64 = AtomicU64::new(0);
    let mut g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
    let m = g.get_or_insert_with(HashMap::new);
    // SAFETY: `queue` is a live PC command queue (the caller's contract).
    let q = unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }?;
    if let std::collections::hash_map::Entry::Vacant(e) = m.entry(queue as usize) {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain D3D12 calls on a live queue and its device.
        let made = unsafe { q.GetDevice(&mut device) }.and_then(|()| match device {
            // SAFETY: as above.
            Some(d) => unsafe { d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) },
            None => Err(windows::core::Error::empty()),
        });
        match made {
            Ok(fence) => {
                e.insert(QueueFence { fence, value: 0 });
            }
            Err(err) => {
                if FAILED.fetch_add(1, Ordering::Relaxed) == 0 {
                    xs_log!("flicker probe: no fence for queue {queue:p} ({err}), its submissions are not watched");
                }
                return None;
            }
        }
    }
    let f = m.get_mut(&(queue as usize))?;
    f.value += 1;
    // SAFETY: both objects are live.
    if let Err(err) = unsafe { q.Signal(&f.fence, f.value) } {
        if FAILED.fetch_add(1, Ordering::Relaxed) == 0 {
            xs_log!("flicker probe: the fence of queue {queue:p} cannot be signalled ({err})");
        }
        return None;
    }
    Some(f.value)
}

/// The wait mode: the CPU waits until `queue` reached `value`, for at most [`WAIT_LIMIT`]. A queue may
/// wait for a fence the title signals from the CPU after this submission, which no wait here can see.
/// So after [`LATE_LIMIT`] late submissions the waits stop.
fn wait_for(queue: usize, value: u64) {
    static LATE: AtomicU64 = AtomicU64::new(0);
    if LATE.load(Ordering::Relaxed) >= LATE_LIMIT {
        return;
    }
    let start = Instant::now();
    let mut turns = 0u32;
    while done(queue) < value {
        if start.elapsed() > WAIT_LIMIT {
            let late = LATE.fetch_add(1, Ordering::Relaxed) + 1;
            xs_log!("flicker probe: a submission on queue {queue:#x} did not finish within {} s, the CPU goes on (late submission {late} of {LATE_LIMIT})", WAIT_LIMIT.as_secs());
            if late == LATE_LIMIT {
                xs_log!("flicker probe: the CPU waits stop, the probe only watches from now on");
            }
            return;
        }
        turns += 1;
        if turns < 64 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_micros(200));
        }
    }
}

/// The extent of the table at root parameter `param` of `layout`, at most [`TABLE_SLOTS`].
fn extent(layout: Option<&crate::rts0::RootLayout>, param: usize) -> u32 {
    let Some(crate::rts0::RootParamKind::Table(ranges)) = layout.and_then(|l| l.params.get(param)).map(|p| &p.kind) else { return TABLE_SLOTS };
    crate::title_heap::table_extent(ranges.iter().map(|r| (r.count, Some(r.offset)))).map_or(TABLE_SLOTS, |n| n.min(TABLE_SLOTS))
}

/// The slot `view_diag` keys the view of host GPU handle `handle` by, 0 when not known.
fn console_slot(handle: u64) -> usize {
    let Some(console) = crate::title_heap::console_handle(handle) else { return 0 };
    let Some(v) = crate::title_heap::visible_for(console) else { return 0 };
    v.host_cpu + (console - v.gpu_start) as usize
}

/// The slots of the resource tables bound for graphics on `list`, and the aligned constant buffer
/// views among them (read from the mirror).
fn table_reads(list: usize) -> (Vec<SlotRead>, Vec<MemRead>) {
    use crate::user_mirror::{self as um, Bind, Slot};
    let (mut slots, mut cbvs) = (Vec::new(), Vec::new());
    if !um::active() {
        return (slots, cbvs);
    }
    let object = um::with_shadow(list, |s| s.signature(Bind::Graphics));
    let Some(sig) = um::signature(object) else { return (slots, cbvs) };
    let count = sig.graphics.placed.len();
    let handles: Vec<Option<u64>> = um::with_shadow(list, |s| (0..count).map(|p| s.table_handle(Bind::Graphics, p)).collect());
    let layout = crate::dxil_override::root_layout(object);
    for (param, placed) in sig.graphics.placed.iter().enumerate() {
        if !placed.is_some_and(|p| p.slot == (Slot::Table { sampler: false })) {
            continue;
        }
        let Some(handle) = handles.get(param).copied().flatten() else { continue };
        let Some((host, inc)) = crate::title_heap::host_slot(handle) else { continue };
        for k in 0..extent(layout.as_ref(), param) {
            let at = host + (k * inc) as usize;
            let Some(print) = print_at(at, inc) else { continue };
            let console = console_slot(handle + u64::from(k * inc));
            let view = if console != 0 { crate::view_diag::view(console) } else { None };
            slots.push(SlotRead { host: at, len: inc, console, res: view.map_or(0, |v| v.res), print, draw: 0, param: param as u16, offset: k as u16 });
            if let Some(v) = view
                && v.va != 0
                && v.va.is_multiple_of(256)
                && matches!(v.kind, "cbv" | "copy cbv")
                && crate::title_mem::writable(v.va as usize, 256)
            {
                cbvs.push(MemRead { va: v.va, len: 256, draw: 0, source: Source::Cbv });
            }
        }
    }
    (slots, cbvs)
}

/// The title ranges of the indices and the vertices draw `d` of `list` reads through the mirror.
fn buffer_reads(list: usize, d: Draw) -> Vec<MemRead> {
    let mut out = Vec::new();
    let (index_view, vertex_views) = crate::draw_diag::buffers(list);
    let mut push = |va: u64, len: u32, source: Source| {
        if len > 0 && crate::title_mem::writable(va as usize, len as usize) {
            out.push(MemRead { va, len, draw: 0, source });
        }
    };
    // (index address, bytes, stride, first index, base vertex)
    let indexed = match d {
        Draw::Plain { .. } => None,
        Draw::Indexed { count, first, base } => index_view.map(|v| {
            let va = u64::from_le_bytes(v[0..8].try_into().expect("eight bytes"));
            let size = u32::from_le_bytes(v[8..12].try_into().expect("four bytes"));
            let stride = if u32::from_le_bytes(v[12..16].try_into().expect("four bytes")) == 42 { 4u32 } else { 2 };
            let start = first.saturating_mul(stride).min(size);
            (va + u64::from(start), count.saturating_mul(stride).min(size - start), stride, base)
        }),
        Draw::IndexedAt { va, format, count } => {
            let stride = if format == 42 { 4u32 } else { 2 };
            Some((va, count.saturating_mul(stride), stride, 0))
        }
    };
    let vertices = match (d, indexed) {
        (Draw::Plain { vertices, first }, _) => (vertices > 0).then(|| (i64::from(first), i64::from(first) + i64::from(vertices) - 1)),
        (_, Some((va, len, stride, base))) if va != 0 && len > 0 => {
            push(va, len.min(MAX_RANGE), Source::Index);
            let span = if len <= MAX_INDICES * stride {
                usize::try_from(va).ok().and_then(|at| read_bytes(at, len as usize)).and_then(|bytes| index_span(&bytes, stride as usize))
            } else {
                None
            };
            span.map(|(lo, hi)| (i64::from(lo) + i64::from(base), i64::from(hi) + i64::from(base)))
        }
        _ => None,
    };
    for (slot, v) in vertex_views {
        let (size, stride) = (v[1] & 0xFFFF_FFFF, v[1] >> 32);
        if stride == 0 {
            continue;
        }
        if let Some((va, len)) = vertex_range(v[0], size, stride, vertices) {
            push(va, len, Source::Vertex(slot));
        }
    }
    out
}

/// Graphics draw `d` is recorded on `list`, after its tables were synced and flushed.
pub(crate) fn draw(list: usize, d: Draw) {
    if !on() {
        return;
    }
    let (slots, mut mems) = table_reads(list);
    mems.extend(buffer_reads(list, d));
    let target = crate::rt_diag::list_state(list);
    let found = with(|c| c.draw(list, target, &slots, &mems));
    report(&found);
}

/// `list` was reset or released.
pub(crate) fn forget(list: usize) {
    if on() {
        with(|c| c.forget(list));
    }
}

/// Before the memory sync of a submission: the title bytes of running submissions that changed.
pub(crate) fn before_sync() -> Vec<Suspect> {
    if !on() {
        return Vec::new();
    }
    with(|c| c.suspects(print_title))
}

/// After the sync, right before the PC queue `queue` runs `lists`: confirms `suspects`, checks the
/// running submissions and the slots of `lists`. The reads to keep until the fence.
pub(crate) fn submitting(queue: *mut c_void, lists: &[usize], suspects: Vec<Suspect>) -> Option<Batch> {
    if !on() {
        return None;
    }
    let present = crate::title_heap::presents();
    let (found, batch) = with(|c| {
        let mut found = c.confirm(&suspects, present, done);
        found.extend(c.check(present, print_at, done));
        let (more, batch) = c.submit(queue as usize, lists, present, print_at, print_title);
        found.extend(more);
        (found, batch)
    });
    report(&found);
    batch
}

/// After the submission on `queue`: signals the probe's fence and keeps `batch` until it is reached.
/// In the wait mode the CPU waits for it.
pub(crate) fn submitted(queue: *mut c_void, batch: Option<Batch>) {
    if !on() {
        return;
    }
    let Some(value) = signal(queue) else { return };
    if let Some(b) = batch {
        with(|c| c.push(b, value));
    }
    if mode() == WAIT {
        wait_for(queue as usize, value);
    }
}

/// A present: checks the running submissions and logs the counts now and then.
pub(crate) fn present() {
    if !on() {
        return;
    }
    let n = crate::title_heap::presents();
    let (found, c) = with(|c| (c.check(n, print_at, done), c.counts));
    report(&found);
    if n.is_multiple_of(REPORT_PRESENTS) {
        xs_log!(
            "flicker probe at present {n}: {} draws, {} slot reads, {} memory reads. {} slots rewritten inside a list, {} slots changed before submission, {} slots rewritten in flight, {} mirror copies in flight. {} submissions, {} done, at most {} in flight, {} reads or submissions dropped",
            c.draws, c.slot_reads, c.mem_reads, c.in_list, c.before_submit, c.slot_in_flight, c.mirror_in_flight, c.batches, c.retired, c.most_in_flight, c.dropped
        );
    }
}

/// Lines logged per kind of finding.
static LINES: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// The first 32 findings of a kind, then each power of two.
fn due(kind: usize) -> bool {
    let n = LINES[kind].fetch_add(1, Ordering::Relaxed) + 1;
    n <= 32 || n.is_power_of_two()
}

/// The view `view_diag` holds for `console` now, or the resource `res` it held, for a line.
fn named(res: usize) -> String {
    if res == 0 {
        return "no resource".into();
    }
    format!("resource {res:#x}")
}

fn now_named(console: usize) -> String {
    match (console != 0).then(|| crate::view_diag::view(console)).flatten() {
        Some(v) if v.res != 0 => format!("resource {:#x} {}x{} format {} ({})", v.res, v.shape.0, v.shape.1, v.shape.2, v.kind),
        Some(v) => format!("{} va {:#x}", v.kind, v.va),
        None => "no view record".into(),
    }
}

fn slot_text(r: &SlotRead, t: Target) -> String {
    format!("host slot {:#x} (table param {} offset {}) of draw {} (target {:#x} pipeline {:#x})", r.host, r.param, r.offset, r.draw, t.0, t.1)
}

fn report(found: &[Finding]) {
    for f in found {
        match f {
            Finding::InList { list, first, later, targets } if due(0) => xs_log!(
                "flicker probe: list {list:#x}: {} and draw {} (target {:#x} pipeline {:#x}) read it with other descriptors ({} then {}, now {}). The GPU gives both draws the later one",
                slot_text(first, targets[0]),
                later.draw,
                targets[1].0,
                targets[1].1,
                named(first.res),
                named(later.res),
                now_named(later.console)
            ),
            Finding::BeforeSubmit { list, read, now, target } if due(1) => xs_log!(
                "flicker probe: list {list:#x}: {} changed before the submission (fingerprint {:#x} to {now:#x}, recorded with {}, now {})",
                slot_text(read, *target),
                read.print,
                named(read.res),
                now_named(read.console)
            ),
            Finding::SlotInFlight { list, read, now, flight, target } if due(2) => xs_log!(
                "flicker probe: list {list:#x}: {} was rewritten while the list ran (fence {} not reached, completed {}, {} presents after the submission, fingerprint {:#x} to {now:#x}, recorded with {}, now {})",
                slot_text(read, *target),
                flight.fence,
                flight.done,
                flight.presents,
                read.print,
                named(read.res),
                now_named(read.console)
            ),
            Finding::MirrorInFlight { list, read, was, now, flight, target } if due(3) => xs_log!(
                "flicker probe: list {list:#x}: {:?} bytes {:#x} ({} bytes) of draw {} (target {:#x} pipeline {:#x}) changed and went into the mirror while the list ran (fence {} not reached, completed {}, {} presents after the submission, fingerprint {was:#x} to {now:#x})",
                read.source,
                read.va,
                read.len,
                read.draw,
                target.0,
                target.1,
                flight.fence,
                flight.done,
                flight.presents
            ),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(host: usize, print: u64) -> SlotRead {
        SlotRead { host, len: 32, console: 0, res: 0, print, draw: 0, param: 1, offset: 0 }
    }

    fn mem(va: u64) -> MemRead {
        MemRead { va, len: 64, draw: 0, source: Source::Vertex(0) }
    }

    #[test]
    fn the_mode_follows_the_variable() {
        assert_eq!(parse_mode(None), OFF);
        assert_eq!(parse_mode(Some("0")), OFF);
        assert_eq!(parse_mode(Some(" ")), OFF);
        assert_eq!(parse_mode(Some("1")), WATCH);
        assert_eq!(parse_mode(Some("wait")), WAIT);
        assert_eq!(parse_mode(Some("WAIT")), WAIT);
    }

    #[test]
    fn fingerprints_tell_bytes_and_lengths_apart() {
        let a = [7u8; 40];
        let mut b = a;
        b[37] = 8;
        assert_eq!(fingerprint(&a), fingerprint(&a.clone()));
        assert_ne!(fingerprint(&a), fingerprint(&b), "a byte past the last whole word");
        assert_ne!(fingerprint(&[0u8; 8]), fingerprint(&[0u8; 16]), "zeros of two lengths");
    }

    #[test]
    fn a_slot_two_draws_of_a_list_read_with_other_descriptors_is_found_at_the_second() {
        let mut c = Core::default();
        assert!(c.draw(1, (0xA, 0xB), &[slot(0x100, 5), slot(0x120, 6)], &[]).is_empty());
        assert!(c.draw(1, (0xA, 0xC), &[slot(0x100, 5)], &[]).is_empty(), "the same descriptor again");
        let found = c.draw(1, (0xA, 0xD), &[slot(0x100, 9)], &[]);
        assert_eq!(found.len(), 1);
        let Finding::InList { first, later, targets, .. } = found[0] else { panic!("{found:?}") };
        assert_eq!((first.draw, later.draw, first.print, later.print), (1, 3, 5, 9));
        assert_eq!(targets, [(0xA, 0xB), (0xA, 0xD)]);
        assert!(c.draw(2, (0, 0), &[slot(0x100, 1)], &[]).is_empty(), "another list");
        assert_eq!((c.counts.in_list, c.counts.slot_reads, c.counts.draws), (1, 3, 4));
    }

    #[test]
    fn a_slot_changed_between_its_draw_and_the_submission_is_found_once() {
        let mut c = Core::default();
        c.draw(1, (0, 0), &[slot(0x100, 5), slot(0x120, 6)], &[]);
        let print = |host: usize, _| Some(if host == 0x100 { 7 } else { 6 });
        let (found, batch) = c.submit(9, &[1], 0, print, |_, _| None);
        assert!(matches!(found[..], [Finding::BeforeSubmit { now: 7, read: SlotRead { host: 0x100, print: 5, .. }, .. }]), "{found:?}");
        assert_eq!(batch.expect("slots to keep").slots.len(), 2);
        let (found, _) = c.submit(9, &[1], 0, print, |_, _| None);
        assert!(found.is_empty(), "a list submitted again starts from what it had at the last submission");
    }

    #[test]
    fn a_slot_rewritten_counts_only_while_its_fence_is_not_reached() {
        let mut c = Core::default();
        c.draw(1, (0, 0), &[slot(0x100, 5)], &[]);
        let (_, batch) = c.submit(9, &[1], 10, |_, _| Some(5), |_, _| None);
        c.push(batch.expect("a slot"), 3);
        assert!(c.check(11, |_, _| Some(5), |_| 2).is_empty(), "unchanged");
        let found = c.check(12, |_, _| Some(8), |_| 2);
        assert!(matches!(found[..], [Finding::SlotInFlight { now: 8, flight: Flight { fence: 3, done: 2, presents: 2 }, .. }]), "{found:?}");
        assert!(c.check(12, |_, _| Some(8), |_| 2).is_empty(), "the same change once");
        assert!(c.check(13, |_, _| Some(4), |_| 3).is_empty(), "the fence is reached: no claim, the submission goes");
        assert_eq!((c.counts.slot_in_flight, c.counts.retired, c.flight.len()), (1, 1, 0));
    }

    #[test]
    fn title_bytes_copied_while_their_list_runs_are_confirmed_after_the_sync() {
        let mut c = Core::default();
        c.draw(1, (0xA, 0xB), &[], &[mem(0x5000), mem(0x6000), mem(0x5000)]);
        assert_eq!(c.counts.mem_reads, 2, "one range once per list");
        let (_, batch) = c.submit(9, &[1], 0, |_, _| None, |va, _| Some(va));
        c.push(batch.expect("memory reads"), 1);
        assert!(c.suspects(|va, _| Some(va)).is_empty(), "fingerprints equal to the addresses: no change");
        let suspects = c.suspects(|va, _| Some(if va == 0x6000 { 1 } else { va }));
        assert_eq!(suspects.len(), 1);
        assert!(c.confirm(&suspects, 4, |_| 1).is_empty(), "the list finished before the sync ended");
        let found = c.confirm(&suspects, 4, |_| 0);
        assert!(matches!(found[..], [Finding::MirrorInFlight { was: 0x6000, now: 1, read: MemRead { va: 0x6000, draw: 1, .. }, target: (0xA, 0xB), .. }]), "{found:?}");
        assert!(c.confirm(&suspects, 4, |_| 0).is_empty(), "counted once");
        assert_eq!(c.counts.mirror_in_flight, 1);
    }

    #[test]
    fn a_reset_list_starts_over_and_running_submissions_keep_their_reads() {
        let mut c = Core::default();
        c.draw(1, (0, 0), &[slot(0x100, 5)], &[]);
        let (_, batch) = c.submit(9, &[1], 0, |_, _| Some(5), |_, _| None);
        c.push(batch.expect("a slot"), 1);
        c.forget(1);
        assert!(c.draw(1, (0, 0), &[slot(0x100, 6)], &[]).is_empty(), "not the same recording");
        assert_eq!(c.lists[&1].draws, 1);
        assert_eq!(c.check(0, |_, _| Some(6), |_| 0).len(), 1, "the running submission still reads the old descriptor");
        let (found, batch) = c.submit(9, &[2], 0, |_, _| Some(6), |_, _| None);
        assert!(found.is_empty() && batch.is_none(), "a list without reads keeps nothing");
    }

    #[test]
    fn the_oldest_submissions_go_past_the_limit() {
        let mut c = Core::default();
        c.draw(1, (0, 0), &[slot(0x100, 5)], &[]);
        for fence in 1..=(MAX_BATCHES as u64 + 2) {
            let (_, batch) = c.submit(9, &[1], 0, |_, _| Some(5), |_, _| None);
            c.push(batch.expect("a slot"), fence);
        }
        assert_eq!((c.flight.len(), c.counts.dropped, c.flight[0].fence), (MAX_BATCHES, 2, 3));
    }

    #[test]
    fn indexed_draws_read_the_vertices_their_indices_name() {
        let bytes: Vec<u8> = [5u16, 9, 7, 6].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(index_span(&bytes, 2), Some((5, 9)));
        let wide: Vec<u8> = [70_000u32, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(index_span(&wide, 4), Some((3, 70_000)));
        assert_eq!(index_span(&[], 2), None);
        assert_eq!(vertex_range(0x1000, 1000, 16, Some((5, 9))), Some((0x1000 + 80, 80)));
        assert_eq!(vertex_range(0x1000, 100, 16, Some((5, 9))), Some((0x1000 + 80, 20)), "cut at the end of the buffer");
        assert_eq!(vertex_range(0x1000, 64, 16, Some((5, 9))), None, "past the buffer");
        assert_eq!(vertex_range(0x1000, 64, 16, Some((-3, 1))), Some((0x1000, 32)), "a negative base vertex starts at 0");
        assert_eq!(vertex_range(0x1000, 1 << 30, 16, None), Some((0x1000, MAX_RANGE)), "unknown vertices: the start of the buffer");
        assert_eq!(vertex_range(0, 64, 16, None), None);
    }

    #[test]
    fn read_bytes_copies_valid_memory_and_rejects_invalid() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let at = data.as_ptr() as usize;
        assert_eq!(read_bytes(at, data.len()), Some(data.to_vec()));
        assert_eq!(print_at(at, data.len() as u32), Some(fingerprint(&data)));
        assert_eq!(read_bytes(0, 8), None);
        assert_eq!(read_bytes(at, 0), None);
        assert_eq!(read_bytes(usize::MAX - 4, 8), None);
    }
}
