//! The GPU side of the device memory window and the translator's service buffers
//! (`window_service` holds the rules).
//!
//! * The window is chosen from the window slots of `title_mem` (`choose_window`) whenever the
//!   generation of `xmem.dll` changes, and [`Snapshot::window`] is what the layer binds: the root UAV at
//!   the base address and the base address in the root CBV. `epoch` counts the changes of the window so
//!   a list sets the base again.
//! * The page table is one upload heap buffer of `TABLE_PAGES` entries that the CPU keeps mapped. It is
//!   filled by asking `title_mem` for the GPU address of every 16 KiB console page (`title_mem::lookup`,
//!   which does not bring pages of commits on demand in: a page the GPU was not given, or whose address
//!   is outside the window, is 0). The table covers the first 16 GiB of title addresses, where `xmem.dll`
//!   puts the title's commits with a console page bit. The first fill is made when the table is first
//!   needed, later ones in a thread when the generation changed, `RESCAN_GIVEN` after pages of commits on
//!   demand reached the GPU, or after `RESCAN` since the last, so a page the title mapped later shows up
//!   within a few seconds.
//! * The three service UAVs are zero filled default heap buffers that stay in the unordered access state.
//!   The shader sets bits in `u0` (pages touched) and `u1` (pages missing, as far as the README says) and
//!   nothing reads them yet.
//!
//! Approximate: only title memory in xmem sections is in the table, not upload memory nor placed
//! resources. The table is rewritten in place while the GPU may read it, entries only change when the
//! title maps pages. The GPU reads through the root UAV by address without a barrier for the section
//! buffers (they are in the common state).

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use crate::log::xs_log;
use crate::window_service::{Extent, PAGE_SHIFT, TABLE_PAGES, Window, choose_window, fill_page_table};

/// Bytes of each service UAV (1 MiB: the bit sets of 1 M pages are 128 KiB each).
const BITS_BYTES: u64 = 1 << 20;
/// How long after a table fill a new one may start when nothing changed in the sections.
const RESCAN: Duration = Duration::from_secs(5);
/// How long after a table fill a new one may start when pages of commits on demand reached the GPU.
const RESCAN_GIVEN: Duration = Duration::from_millis(250);

/// What a list binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// The window, `None` when there is no section buffer (the base is then 0).
    pub window: Option<Window>,
    /// GPU address of the page table.
    pub table_gpu: u64,
    /// GPU addresses of the three service UAVs.
    pub bits_gpu: [u64; 3],
    /// Changes whenever the window changes, starts at 1.
    pub epoch: u64,
}

struct State {
    snapshot: Snapshot,
    table_cpu: usize,
    generation: Option<u32>,
    last_scan: Instant,
    /// `title_mem::given_pages` at the last fill.
    given: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static FAILED: AtomicBool = AtomicBool::new(false);
static SCANNING: AtomicBool = AtomicBool::new(false);

fn buffer(device: &ID3D12Device, heap: D3D12_HEAP_TYPE, bytes: u64, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> windows::core::Result<ID3D12Resource> {
    let props = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: plain description data, `res` is a valid output.
    unsafe { device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res) }?;
    res.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))
}

/// Makes the page table and the three service buffers on the device of `list`. They live for the process.
///
/// # Safety
/// `list` is a live PC command list.
unsafe fn create(list: *mut c_void) -> windows::core::Result<State> {
    // SAFETY: by the contract above, nothing is retained from the borrow.
    let list = unsafe { ID3D12GraphicsCommandList::from_raw_borrowed(&list) }.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4003_u32 as i32)))?;
    // SAFETY: a live list.
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: a live list, `device` is a valid output.
    unsafe { list.GetDevice(&mut device) }?;
    let device = device.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))?;
    let bytes = (TABLE_PAGES * 8) as u64;
    let table = buffer(&device, D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
    let mut cpu: *mut c_void = std::ptr::null_mut();
    // SAFETY: an upload buffer, mapped for the life of the process.
    unsafe { table.Map(0, None, Some(&mut cpu)) }?;
    // SAFETY: the mapping is `bytes` long.
    unsafe { std::ptr::write_bytes(cpu.cast::<u8>(), 0, bytes as usize) };
    // SAFETY: a live resource.
    let table_gpu = unsafe { table.GetGPUVirtualAddress() };
    let mut bits_gpu = [0u64; 3];
    for slot in &mut bits_gpu {
        let b = buffer(&device, D3D12_HEAP_TYPE_DEFAULT, BITS_BYTES, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
        // SAFETY: a live resource.
        *slot = unsafe { b.GetGPUVirtualAddress() };
        crate::mem_owners::add(
            crate::mem_owners::Kind::ServiceBuffer,
            crate::mem_owners::Segment::Local,
            1,
            BITS_BYTES,
        );
        std::mem::forget(b);
    }
    crate::mem_owners::add(
        crate::mem_owners::Kind::PageTable,
        crate::mem_owners::Segment::Unknown,
        1,
        bytes,
    );
    std::mem::forget(table);
    xs_log!("window: page table of {TABLE_PAGES} pages at GPU {table_gpu:#x}, service buffers at {:#x} {:#x} {:#x}", bits_gpu[0], bits_gpu[1], bits_gpu[2]);
    Ok(State {
        snapshot: Snapshot { window: None, table_gpu, bits_gpu, epoch: 0 },
        table_cpu: cpu as usize,
        generation: None,
        last_scan: Instant::now(),
        given: 0,
    })
}

/// Fills the page table at `cpu` from the sections of `title_mem`.
fn scan(cpu: usize, window: Option<Window>) {
    let mut table = vec![0u64; TABLE_PAGES];
    let stats = fill_page_table(&mut table, window.as_ref(), |va, len| crate::title_mem::lookup(va, len).map(|s| s.gpu));
    // SAFETY: the table buffer is `TABLE_PAGES * 8` bytes, mapped and never released.
    unsafe { std::ptr::copy_nonoverlapping(table.as_ptr(), cpu as *mut u64, TABLE_PAGES) };
    xs_log!("window: page table filled, {} pages mapped, {} outside the window and left out", stats.mapped, stats.outside);
}

/// Starts a fill in a thread unless one runs.
fn scan_later(cpu: usize, window: Option<Window>) {
    if SCANNING.swap(true, Ordering::AcqRel) {
        return;
    }
    let started = std::thread::Builder::new().name("window-page-table".into()).spawn(move || {
        scan(cpu, window);
        SCANNING.store(false, Ordering::Release);
    });
    if started.is_err() {
        SCANNING.store(false, Ordering::Release);
    }
}

/// The window, the table and the buffers to bind now, creating them on first use (on the device of `list`)
/// and following the sections of `xmem.dll`. `None` when they cannot be made (logged once).
///
/// # Safety
/// `list` is a live PC command list.
pub unsafe fn current(list: *mut c_void) -> Option<Snapshot> {
    if FAILED.load(Ordering::Acquire) {
        return None;
    }
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let first = guard.is_none();
    if first {
        // SAFETY: by the contract above.
        match unsafe { create(list) } {
            Ok(state) => *guard = Some(state),
            Err(e) => {
                FAILED.store(true, Ordering::Release);
                xs_log!("window: the page table and service buffers cannot be made ({e}), the parameters are bound as null");
                return None;
            }
        }
    }
    let st = guard.as_mut()?;
    let generation = crate::title_mem::generation();
    let given = crate::title_mem::given_pages();
    let mut rescan = st.last_scan.elapsed() >= RESCAN || (given != st.given && st.last_scan.elapsed() >= RESCAN_GIVEN);
    if generation != st.generation || first {
        st.generation = generation;
        rescan = true;
        // Slots of title memory the page table covers come first: the table only maps console pages below
        // 16 GiB, so a window over slots above it gives the translated shaders nothing by address (a
        // run of 24 Nov: a 3.7 GiB reservation high up took the window from the arenas below 16 GiB).
        let slots = crate::title_mem::extents().map(|(_, e)| e).unwrap_or_default();
        let reach = (TABLE_PAGES as u64) << PAGE_SHIFT;
        let covered: Vec<Extent> = slots.iter().filter(|e| e.va.saturating_add(e.bytes) <= reach).map(|e| Extent { gpu: e.gpu, bytes: e.bytes }).collect();
        let extents: Vec<Extent> = if covered.is_empty() { slots.iter().map(|e| Extent { gpu: e.gpu, bytes: e.bytes }).collect() } else { covered };
        let chosen = choose_window(&extents);
        let window = chosen.map(|c| c.window);
        if window != st.snapshot.window || st.snapshot.epoch == 0 {
            st.snapshot.epoch += 1;
            st.snapshot.window = window;
            match chosen {
                Some(c) => xs_log!(
                    "window: base {:#x}, end {:#x}, {} of {} section buffers inside 4 GiB above the base{}",
                    c.window.base,
                    c.window.end,
                    c.covered,
                    c.total,
                    if c.covered < c.total { " (the others are not reachable by translated shaders)" } else { "" }
                ),
                None => xs_log!("window: no section buffer is open, the base is 0 and translated device addresses read nothing"),
            }
        }
        if first {
            // The shader needs the first table at once.
            scan(st.table_cpu, st.snapshot.window);
            st.last_scan = Instant::now();
            rescan = false;
        }
    }
    if rescan {
        st.given = given;
        st.last_scan = Instant::now();
        scan_later(st.table_cpu, st.snapshot.window);
    }
    Some(st.snapshot)
}

/// The bytes of the page table (upload heap) and of the three service buffers (default heap), 0 before
/// they are made. For the memory sampler.
pub fn bytes() -> (u64, u64) {
    if STATE.lock().unwrap_or_else(|e| e.into_inner()).is_some() { ((TABLE_PAGES * 8) as u64, 3 * BITS_BYTES) } else { (0, 0) }
}

/// Whether the PC address `addr` (a root view the mirror carries) is outside the window, logged a few times.
/// False while there is no window state.
pub fn outside_window(addr: u64) -> bool {
    static N: AtomicI32 = AtomicI32::new(0);
    let guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(st) = guard.as_ref() else { return false };
    let outside = st.snapshot.window.is_none_or(|w| !w.contains(addr, 4));
    if outside && N.fetch_add(1, Ordering::Relaxed) < 8 {
        xs_log!("window: root view address {addr:#x} is outside the window {:?}, a translated shader cannot reach it", st.snapshot.window);
    }
    outside
}
