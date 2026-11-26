//! Atomic accounting for the memory owners the D3D12.X layer creates.
//!
//! The sampler reads a fixed number of atomics; it never walks a resource table. A resource's
//! allocation size is recorded once at creation and removed once at its final release. Sizes are
//! bytes here and are converted to MiB only when the diagnostic line is formatted.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const MIB: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum Kind {
    ViewTexture,
    ViewHeap,
    DetileStaging,
    DetileParams,
    DetileTexture,
    CommittedRenderTarget,
    CommittedDepth,
    CommittedSampled,
    CommittedOther,
    PlacedResource,
    GpuCopyScratch,
    TitleMirrorHeap,
    MirrorTile,
    UploadChunk,
    Readback,
    WindowReserved,
    WindowCommitted,
    DescriptorHeap,
    DescriptorPageMap,
    TitleDescriptorMemory,
    ViewShadow,
    PageTable,
    ServiceBuffer,
}

const KIND_COUNT: usize = Kind::ServiceBuffer as usize + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum Segment {
    Local,
    System,
    Unknown,
}

const SEGMENT_COUNT: usize = Segment::Unknown as usize + 1;
const SLOT_COUNT: usize = KIND_COUNT * SEGMENT_COUNT;

impl Kind {
    /// Informational sizes live inside another owner and must not be summed a second time.
    fn additive(self) -> bool {
        !matches!(
            self,
            Self::ViewTexture
                | Self::MirrorTile
                | Self::WindowReserved
                | Self::DetileTexture
                | Self::PlacedResource
        )
    }
}

#[repr(align(64))]
struct Slot {
    count: AtomicU64,
    bytes: AtomicU64,
}

impl Slot {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }
}

struct Accounting {
    slots: [Slot; SLOT_COUNT],
    view_cap_bytes: AtomicU64,
    view_heaps_evicted: AtomicU64,
}

impl Accounting {
    const fn new() -> Self {
        Self {
            slots: [const { Slot::new() }; SLOT_COUNT],
            view_cap_bytes: AtomicU64::new(0),
            view_heaps_evicted: AtomicU64::new(0),
        }
    }

    fn slot(&self, kind: Kind, segment: Segment) -> &Slot {
        &self.slots[kind as usize * SEGMENT_COUNT + segment as usize]
    }

    fn add(&self, kind: Kind, segment: Segment, count: u64, bytes: u64) {
        let slot = self.slot(kind, segment);
        slot.count.fetch_add(count, Ordering::Relaxed);
        slot.bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn remove(&self, kind: Kind, segment: Segment, count: u64, bytes: u64) {
        let slot = self.slot(kind, segment);
        if count != 0 {
            subtract(&slot.count, count);
        }
        if bytes != 0 {
            subtract(&slot.bytes, bytes);
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            slots: std::array::from_fn(|i| OwnerValue {
                count: self.slots[i].count.load(Ordering::Relaxed),
                bytes: self.slots[i].bytes.load(Ordering::Relaxed),
            }),
            view_cap_bytes: self.view_cap_bytes.load(Ordering::Relaxed),
            view_heaps_evicted: self.view_heaps_evicted.load(Ordering::Relaxed),
            view_heaps_empty: 0,
        }
    }
}

fn subtract(value: &AtomicU64, amount: u64) {
    // A release must not wrap a counter if a bookkeeping path is reached twice. Ignoring the
    // failed update keeps the diagnostic finite; the unit test covers the guard.
    let _ = value.try_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
        old.checked_sub(amount)
    });
}

static OWNERS: Accounting = Accounting::new();

pub(crate) fn add(kind: Kind, segment: Segment, count: u64, bytes: u64) {
    #[cfg(all(windows, not(test)))]
    trace_sampler::maybe_start();
    OWNERS.add(kind, segment, count, bytes);
}

pub(crate) fn remove(kind: Kind, segment: Segment, count: u64, bytes: u64) {
    OWNERS.remove(kind, segment, count, bytes);
}

pub(crate) fn set_view_cap(bytes: u64) {
    OWNERS.view_cap_bytes.store(bytes, Ordering::Relaxed);
}

pub(crate) fn note_view_heaps_evicted(count: u64) {
    OWNERS.view_heaps_evicted.fetch_add(count, Ordering::Relaxed);
}

pub(crate) fn snapshot() -> Snapshot {
    OWNERS.snapshot()
}

/// One additive allocation carried with an object until its actual COM release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Charge {
    pub(crate) kind: Kind,
    pub(crate) segment: Segment,
    pub(crate) bytes: u64,
}

pub(crate) fn add_charge(charge: Charge) {
    add(charge.kind, charge.segment, 1, charge.bytes);
}

pub(crate) fn remove_charge(charge: Charge) {
    remove(charge.kind, charge.segment, 1, charge.bytes);
}

/// Classifies a committed resource for the diagnostic. Buffers and unclassifiable textures are
/// `other`; a texture that permits shader reads and is neither a render target nor depth is
/// `sampled`. The numeric values are the public `D3D12_RESOURCE_DIMENSION` and flag bits.
pub(crate) fn committed_kind(dimension: i32, flags: i32) -> Kind {
    if !(2..=4).contains(&dimension) {
        return Kind::CommittedOther;
    }
    if flags & 0x1 != 0 {
        Kind::CommittedRenderTarget
    } else if flags & 0x2 != 0 {
        Kind::CommittedDepth
    } else if flags & 0x8 == 0 {
        Kind::CommittedSampled
    } else {
        Kind::CommittedOther
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OwnerValue {
    pub(crate) count: u64,
    pub(crate) bytes: u64,
}

impl OwnerValue {
    fn mib(self) -> u64 {
        self.bytes / MIB
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Snapshot {
    slots: [OwnerValue; SLOT_COUNT],
    view_cap_bytes: u64,
    view_heaps_evicted: u64,
    /// View heaps holding no live texture; filled in by the sampler from `view_heap`.
    view_heaps_empty: u64,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self { slots: [OwnerValue::default(); SLOT_COUNT], view_cap_bytes: 0, view_heaps_evicted: 0, view_heaps_empty: 0 }
    }
}

impl Snapshot {
    pub(crate) fn with_empty_view_heaps(mut self, empty: usize) -> Self {
        self.view_heaps_empty = empty as u64;
        self
    }

    fn value(&self, kind: Kind, segment: Segment) -> OwnerValue {
        self.slots[kind as usize * SEGMENT_COUNT + segment as usize]
    }

    fn total(&self, kind: Kind) -> OwnerValue {
        let mut total = OwnerValue::default();
        for segment in [Segment::Local, Segment::System, Segment::Unknown] {
            let value = self.value(kind, segment);
            total.count += value.count;
            total.bytes += value.bytes;
        }
        total
    }

    fn accounted(&self, segment: Segment) -> u64 {
        let mut bytes = 0;
        for kind in [
            Kind::ViewTexture,
            Kind::ViewHeap,
            Kind::DetileStaging,
            Kind::DetileParams,
            Kind::DetileTexture,
            Kind::CommittedRenderTarget,
            Kind::CommittedDepth,
            Kind::CommittedSampled,
            Kind::CommittedOther,
            Kind::PlacedResource,
            Kind::GpuCopyScratch,
            Kind::TitleMirrorHeap,
            Kind::MirrorTile,
            Kind::UploadChunk,
            Kind::Readback,
            Kind::WindowReserved,
            Kind::WindowCommitted,
            Kind::DescriptorHeap,
            Kind::DescriptorPageMap,
            Kind::TitleDescriptorMemory,
            Kind::ViewShadow,
            Kind::PageTable,
            Kind::ServiceBuffer,
        ] {
            if kind.additive() {
                bytes += self.value(kind, segment).bytes;
            }
        }
        bytes
    }

    fn local_accounted(&self) -> u64 {
        self.accounted(Segment::Local)
    }
}

fn push_item(out: &mut String, label: &str, details: impl std::fmt::Display) {
    write!(out, " {label} {details};").expect("writing to a String cannot fail");
}

fn value_text(value: OwnerValue) -> String {
    format!("{}/{}", value.count, value.mib())
}

fn segment_committed(snapshot: &Snapshot, segment: Segment) -> String {
    let categories = [
        (Kind::CommittedRenderTarget, "RT"),
        (Kind::CommittedDepth, "depth"),
        (Kind::CommittedSampled, "sampled"),
        (Kind::CommittedOther, "other"),
    ];
    categories
        .into_iter()
        .map(|(kind, label)| format!("{label} {}", value_text(snapshot.value(kind, segment))))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn format(snapshot: &Snapshot, local_usage: Option<u64>) -> String {
    let mut out = String::with_capacity(768);
    out.push_str("memory by owner (count/MiB): GPU local:");
    let textures = snapshot.total(Kind::ViewTexture);
    let heaps = snapshot.total(Kind::ViewHeap);
    let cap = if snapshot.view_cap_bytes == 0 {
        "unlimited".to_owned()
    } else {
        format!("{}", snapshot.view_cap_bytes / MIB)
    };
    push_item(
        &mut out,
        "view cache",
        format!(
            "{}/{} textures in {}/{} heaps (cap {cap}, {} empty, {} evicted)",
            textures.count,
            textures.mib(),
            heaps.count,
            heaps.mib(),
            snapshot.view_heaps_empty,
            snapshot.view_heaps_evicted
        ),
    );
    for (kind, label) in [
        (Kind::DetileStaging, "detile staging"),
        (Kind::DetileTexture, "detiled textures"),
    ] {
        push_item(&mut out, label, value_text(snapshot.total(kind)));
    }
    push_item(
        &mut out,
        "committed textures",
        segment_committed(snapshot, Segment::Local),
    );
    push_item(
        &mut out,
        "placed resources",
        value_text(snapshot.value(Kind::PlacedResource, Segment::Local)),
    );
    push_item(
        &mut out,
        "GPU scratch",
        value_text(snapshot.total(Kind::GpuCopyScratch)),
    );
    push_item(
        &mut out,
        "descriptor heaps",
        value_text(snapshot.value(Kind::DescriptorHeap, Segment::Local)),
    );
    push_item(
        &mut out,
        "service buffers",
        value_text(snapshot.value(Kind::ServiceBuffer, Segment::Local)),
    );
    if let Some(usage) = local_usage {
        push_item(
            &mut out,
            "not accounted",
            format!("{}", usage.saturating_sub(snapshot.local_accounted()) / MIB),
        );
    }

    out += " system:";
    let mirror = snapshot.total(Kind::TitleMirrorHeap);
    let tiles = snapshot.total(Kind::MirrorTile);
    push_item(
        &mut out,
        "title mirror",
        format!(
            "{} heaps/{} ({} tiles/{} inside)",
            mirror.count,
            mirror.mib(),
            tiles.count,
            tiles.mib()
        ),
    );
    push_item(
        &mut out,
        "committed textures",
        segment_committed(snapshot, Segment::System),
    );
    push_item(
        &mut out,
        "readback",
        value_text(snapshot.total(Kind::Readback)),
    );
    let reserved = snapshot.total(Kind::WindowReserved);
    let committed = snapshot.total(Kind::WindowCommitted);
    push_item(
        &mut out,
        "windows",
        format!(
            "{}/{} reserved, {} committed",
            reserved.count,
            reserved.mib(),
            committed.mib()
        ),
    );
    for (kind, label) in [
        (Kind::DescriptorHeap, "descriptor heaps"),
        (Kind::TitleDescriptorMemory, "title descriptor RAM"),
    ] {
        push_item(
            &mut out,
            label,
            value_text(snapshot.value(kind, Segment::System)),
        );
    }
    push_item(
        &mut out,
        "placed aliases",
        value_text(snapshot.value(Kind::PlacedResource, Segment::System)),
    );

    out += " segment unknown:";
    for (kind, label) in [
        (Kind::UploadChunk, "upload chunks/rings"),
        (Kind::DetileParams, "detile params"),
        (Kind::DescriptorPageMap, "descriptor maps"),
        (Kind::ViewShadow, "view shadow"),
        (Kind::PageTable, "page table"),
    ] {
        push_item(&mut out, label, value_text(snapshot.total(kind)));
    }
    push_item(
        &mut out,
        "committed textures",
        segment_committed(snapshot, Segment::Unknown),
    );
    push_item(
        &mut out,
        "placed resources",
        value_text(snapshot.value(Kind::PlacedResource, Segment::Unknown)),
    );
    out
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ProcessRamSample {
    working_set_bytes: u64,
    private_commit_bytes: u64,
    avail_phys_bytes: u64,
    page_fault_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DstorageMetricsSnapshot {
    requests: u64,
    bytes_read: u64,
    active_read_ns: u64,
    decode_ns: u64,
    write_ns: u64,
    in_flight: u64,
    batches: u64,
}

fn emit_trace_counters(
    snapshot: &Snapshot,
    ram: Option<ProcessRamSample>,
    prev_faults: &mut Option<u32>,
    elapsed_s: f64,
    vram_local_mib: Option<f64>,
    vram_nonlocal_mib: Option<f64>,
    dstorage: Option<DstorageMetricsSnapshot>,
) {
    let mib_f = |b: u64| b as f64 / MIB as f64;
    if let Some(r) = ram {
        decant_trace::record_counter("WorkingSetMiB", mib_f(r.working_set_bytes));
        decant_trace::record_counter("PrivateCommitMiB", mib_f(r.private_commit_bytes));
        if r.avail_phys_bytes > 0 {
            decant_trace::record_counter("AvailRamMiB", mib_f(r.avail_phys_bytes));
        }
        let faults_per_sec = if let Some(prev) = *prev_faults
            && elapsed_s > 0.0
        {
            f64::from(r.page_fault_count.saturating_sub(prev)) / elapsed_s
        } else {
            0.0
        };
        *prev_faults = Some(r.page_fault_count);
        decant_trace::record_counter("PageFaultsPerSec", faults_per_sec);
    }
    if let Some(v) = vram_local_mib {
        decant_trace::record_counter("VramLocalMiB", v);
    }
    if let Some(v) = vram_nonlocal_mib {
        decant_trace::record_counter("VramNonLocalMiB", v);
    }
    if let Some(ds) = dstorage {
        let total_mib = mib_f(ds.bytes_read);
        let active_read_mb_s = if ds.active_read_ns > 0 { total_mib / (ds.active_read_ns as f64 / 1_000_000_000.0) } else { 0.0 };
        let decode_mb_s = if ds.decode_ns > 0 { total_mib / (ds.decode_ns as f64 / 1_000_000_000.0) } else { 0.0 };
        decant_trace::record_counter("dstorage_active_read_mb_s", active_read_mb_s);
        decant_trace::record_counter("dstorage_decode_mb_s", decode_mb_s);
        decant_trace::record_counter("dstorage_total_read_mib", total_mib);
        decant_trace::record_counter("dstorage_in_flight", ds.in_flight as f64);
    }
    decant_trace::record_counter("OwnerViewTexturesMiB", mib_f(snapshot.total(Kind::ViewTexture).bytes));
    decant_trace::record_counter("OwnerViewHeapsMiB", mib_f(snapshot.total(Kind::ViewHeap).bytes));
    decant_trace::record_counter("OwnerTitleMirrorMiB", mib_f(snapshot.total(Kind::TitleMirrorHeap).bytes));
    decant_trace::record_counter("OwnerUploadMiB", mib_f(snapshot.total(Kind::UploadChunk).bytes));
    decant_trace::record_counter("OwnerLocalAccountedMiB", mib_f(snapshot.local_accounted()));
}

#[cfg(all(windows, not(test)))]
#[allow(unsafe_code)]
mod trace_sampler {
    use super::{DstorageMetricsSnapshot, MIB, ProcessRamSample, emit_trace_counters, snapshot};
    use std::sync::Once;
    use std::time::{Duration, Instant};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3, IDXGIFactory4,
    };
    use windows::core::Interface;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    type GetProcessMemoryInfoFn = unsafe extern "system" fn(HANDLE, *mut ProcessMemoryCountersEx, u32) -> i32;
    type DecantDstorageMetricsFn = unsafe extern "C" fn(*mut DstorageMetricsSnapshot) -> u32;

    fn query_ram() -> Option<ProcessRamSample> {
        static FN: std::sync::OnceLock<Option<GetProcessMemoryInfoFn>> = std::sync::OnceLock::new();
        let func = (*FN.get_or_init(|| unsafe {
            let k32 = GetModuleHandleA(c"kernel32.dll".as_ptr().cast());
            let proc = if !k32.is_null() {
                GetProcAddress(k32, c"K32GetProcessMemoryInfo".as_ptr().cast())
            } else {
                None
            };
            proc.map(|p| std::mem::transmute::<unsafe extern "system" fn() -> isize, GetProcessMemoryInfoFn>(p))
        }))?;
        let mut pmc = ProcessMemoryCountersEx {
            cb: std::mem::size_of::<ProcessMemoryCountersEx>() as u32,
            ..Default::default()
        };
        // SAFETY: valid handle and buffer size.
        let ok = unsafe { func(GetCurrentProcess(), &mut pmc, pmc.cb) };
        if ok == 0 {
            return None;
        }
        let mut mem_status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: valid `MEMORYSTATUSEX` struct with `dwLength` initialized.
        let avail_phys_bytes = if unsafe { GlobalMemoryStatusEx(&mut mem_status) } != 0 {
            mem_status.ullAvailPhys
        } else {
            0
        };
        Some(ProcessRamSample {
            working_set_bytes: pmc.working_set_size as u64,
            private_commit_bytes: pmc.private_usage as u64,
            avail_phys_bytes,
            page_fault_count: pmc.page_fault_count,
        })
    }

    fn query_dstorage() -> Option<DstorageMetricsSnapshot> {
        // SAFETY: queries an already-loaded `dstorage_xs.dll` if present in the process.
        let func = unsafe {
            let module = GetModuleHandleA(c"dstorage_xs.dll".as_ptr().cast());
            if module.is_null() {
                return None;
            }
            let proc = GetProcAddress(module, c"DecantDstorageMetrics".as_ptr().cast())?;
            std::mem::transmute::<unsafe extern "system" fn() -> isize, DecantDstorageMetricsFn>(proc)
        };
        let mut out = DstorageMetricsSnapshot::default();
        // SAFETY: `out` is a valid `#[repr(C)]` `DstorageMetricsSnapshot`.
        (unsafe { func(&mut out) } != 0).then_some(out)
    }

    fn query_vram(adapter: Option<&IDXGIAdapter3>) -> (Option<f64>, Option<f64>) {
        let Some(a) = adapter else {
            return (None, None);
        };
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: live adapter and valid out structs.
        let l = unsafe { a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local) }
            .ok()
            .map(|()| local.CurrentUsage as f64 / MIB as f64);
        let nl = unsafe { a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut non_local) }
            .ok()
            .map(|()| non_local.CurrentUsage as f64 / MIB as f64);
        (l, nl)
    }

    pub fn maybe_start() {
        if !decant_trace::enabled() {
            return;
        }
        static STARTED: Once = Once::new();
        STARTED.call_once(|| {
            let _ = std::thread::Builder::new().name("d3d12xs-trace-counters".into()).spawn(|| {
                let adapter: Option<IDXGIAdapter3> = unsafe {
                    CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                        .ok()
                        .and_then(|f| f.EnumAdapters1(0).ok())
                        .and_then(|a| a.cast::<IDXGIAdapter3>().ok())
                };
                let mut prev_faults = None;
                let mut last = Instant::now();
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    let now = Instant::now();
                    let dt = now.duration_since(last).as_secs_f64();
                    last = now;
                    let (vram_local, vram_nonlocal) = query_vram(adapter.as_ref());
                    emit_trace_counters(&snapshot(), query_ram(), &mut prev_faults, dt, vram_local, vram_nonlocal, query_dstorage());
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Snapshot {
        let mut snapshot = Snapshot::default();
        snapshot.slots[Kind::ViewTexture as usize * SEGMENT_COUNT + Segment::Local as usize] =
            OwnerValue {
                count: 7,
                bytes: 700 * MIB,
            };
        snapshot.slots[Kind::ViewHeap as usize * SEGMENT_COUNT + Segment::Local as usize] =
            OwnerValue {
                count: 3,
                bytes: 1024 * MIB,
            };
        snapshot.slots[Kind::TitleMirrorHeap as usize * SEGMENT_COUNT + Segment::System as usize] =
            OwnerValue {
                count: 2,
                bytes: 512 * MIB,
            };
        snapshot.slots[Kind::MirrorTile as usize * SEGMENT_COUNT + Segment::System as usize] =
            OwnerValue {
                count: 8,
                bytes: 512 * MIB,
            };
        snapshot.view_cap_bytes = 14 * 1024 * MIB;
        snapshot.view_heaps_evicted = 4;
        snapshot
    }

    #[test]
    fn additive_sums_do_not_count_nested_sizes_twice() {
        let accounting = Accounting::new();
        accounting.add(Kind::ViewHeap, Segment::Local, 1, 64 * MIB);
        accounting.add(Kind::ViewTexture, Segment::Local, 2, 48 * MIB);
        accounting.add(Kind::DetileTexture, Segment::Local, 2, 48 * MIB);
        accounting.add(Kind::PlacedResource, Segment::Local, 2, 48 * MIB);
        accounting.add(Kind::TitleMirrorHeap, Segment::System, 1, 256 * MIB);
        accounting.add(Kind::MirrorTile, Segment::System, 4, 256 * MIB);
        accounting.add(Kind::UploadChunk, Segment::Unknown, 1, 16 * MIB);
        let snapshot = accounting.snapshot();
        assert_eq!(snapshot.local_accounted(), 64 * MIB);
        assert_eq!(snapshot.accounted(Segment::System), 256 * MIB);
        assert_eq!(snapshot.total(Kind::ViewTexture).bytes, 48 * MIB);
        assert_eq!(snapshot.total(Kind::MirrorTile).bytes, 256 * MIB);
    }

    #[test]
    fn removal_is_balanced_and_cannot_wrap() {
        let accounting = Accounting::new();
        accounting.add(Kind::Readback, Segment::System, 1, 8 * MIB);
        accounting.remove(Kind::Readback, Segment::System, 1, 8 * MIB);
        accounting.remove(Kind::Readback, Segment::System, 1, 8 * MIB);
        let value = accounting.snapshot().total(Kind::Readback);
        assert_eq!(value, OwnerValue::default());
    }

    #[test]
    fn heap_and_committed_resource_classification_is_pure() {
        assert_eq!(committed_kind(1, 0), Kind::CommittedOther);
        assert_eq!(committed_kind(3, 0x1), Kind::CommittedRenderTarget);
        assert_eq!(committed_kind(3, 0x2), Kind::CommittedDepth);
        assert_eq!(committed_kind(3, 0), Kind::CommittedSampled);
        assert_eq!(committed_kind(3, 0x8), Kind::CommittedOther);
    }

    #[test]
    fn formatting_names_segments_and_subtracts_only_local_owners() {
        let line = format(&sample().with_empty_view_heaps(1), Some(20 * 1024 * MIB));
        assert!(line.starts_with("memory by owner (count/MiB): GPU local:"));
        assert!(line.contains("view cache 7/700 textures in 3/1024 heaps"));
        assert!(line.contains("(cap 14336, 1 empty, 4 evicted)"));
        assert!(line.contains("not accounted 19456;"), "{line}");
        assert!(line.contains("; system: title mirror 2 heaps/512 (8 tiles/512 inside);"));
        assert!(line.contains("; segment unknown: upload chunks/rings 0/0;"));
    }

    #[test]
    fn formatting_without_video_usage_omits_the_residual() {
        let line = format(&sample(), None);
        assert!(!line.contains("not accounted"));
    }

    #[test]
    fn a_typical_line_fits_one_log_line() {
        let line = format(&sample(), Some(20 * 1024 * MIB));
        assert!(line.len() < 1000, "owner line was {} bytes", line.len());
    }

    #[test]
    fn slots_are_cache_line_aligned_and_zero_remove_is_noop() {
        assert_eq!(std::mem::align_of::<Slot>(), 64);
        assert_eq!(std::mem::size_of::<Slot>(), 64);
        let accounting = Accounting::new();
        accounting.add(Kind::Readback, Segment::System, 2, 16 * MIB);
        accounting.remove(Kind::Readback, Segment::System, 0, 0);
        assert_eq!(accounting.snapshot().total(Kind::Readback), OwnerValue { count: 2, bytes: 16 * MIB });
    }

    #[test]
    fn trace_counter_emission_updates_prev_faults_safely() {
        let mut prev = None;
        let snap = sample();
        emit_trace_counters(
            &snap,
            Some(ProcessRamSample { working_set_bytes: 1024 * MIB, private_commit_bytes: 2048 * MIB, avail_phys_bytes: 16384 * MIB, page_fault_count: 500 }),
            &mut prev,
            1.0,
            Some(4000.0),
            Some(500.0),
            Some(DstorageMetricsSnapshot { requests: 10, bytes_read: 64 * MIB, active_read_ns: 500_000_000, decode_ns: 250_000_000, write_ns: 100_000_000, in_flight: 2, batches: 3 }),
        );
        assert_eq!(prev, Some(500));
        emit_trace_counters(
            &snap,
            Some(ProcessRamSample { working_set_bytes: 1024 * MIB, private_commit_bytes: 2048 * MIB, avail_phys_bytes: 16000 * MIB, page_fault_count: 750 }),
            &mut prev,
            1.0,
            Some(4100.0),
            Some(500.0),
            None,
        );
        assert_eq!(prev, Some(750));
    }
}
