//! The built in present, used when no presentation layer installed a hook. A console title
//! renders into its own back buffers and hands them to `PresentX`. Here the first resource of
//! plane 0 is copied into a DXGI swap chain on the title's own top level window, on the title's
//! queue, then presented without waiting for vblank (the frame clock in `frame` paces the title).
//!
//! Two back buffers, one command allocator each, and a fence: before a buffer is recorded again
//! the copy that last used it has finished. `DECANT_D3D12X_PRESENT=0` turns it off.
//!
//! `DECANT_D3D12X_PRESENT_PROBE` (see `rt_diag`) reads pixels back from the presented resource,
//! and from the most drawn other render target in `target` mode, for diagnosis. When
//! `DECANT_D3D12X_PROBE_DUMP=<dir>` is also set, each readback that probe already takes is written
//! into `<dir>` as an uncompressed BMP, `frame<n>-presented.bmp` / `frame<n>-target.bmp`, at most
//! one pair every 8 samples and at most 40 files per run.

#![allow(unsafe_code)]

use crate::log::xs_log;
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Foundation::{HWND, LPARAM, RECT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::GetCurrentProcessId;
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONEAREST};
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, EnumWindows, GetClientRect, GetWindowLongPtrW, GetWindowRect, GetWindowThreadProcessId, IsWindowVisible, SetWindowLongPtrW, SetWindowPos, GWL_EXSTYLE,
    GWL_STYLE, SWP_FRAMECHANGED, SWP_NOZORDER, SWP_SHOWWINDOW, WINDOW_EX_STYLE, WINDOW_STYLE, WS_OVERLAPPEDWINDOW, WS_POPUP, SW_RESTORE, ShowWindow, IsZoomed,
};
use windows::core::{BOOL, Interface};

/// The swap chain format for a back buffer of format `f`, `None` when it cannot be shown.
pub fn swap_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match f {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB | DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB | DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

fn enabled() -> bool {
    std::env::var_os("DECANT_D3D12X_PRESENT").is_none_or(|v| v != "0")
}

struct Presenter {
    queue: ID3D12CommandQueue,
    swap: IDXGISwapChain3,
    size: (u32, u32, DXGI_FORMAT),
    allocators: [ID3D12CommandAllocator; 2],
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    /// The fence value that marks the last copy into each back buffer.
    marks: [u64; 2],
    next: u64,
    /// `DECANT_D3D12X_PRESENT_PROBE`: a readback buffer for the presented resource and its layout.
    probe: Option<(crate::mem_resource::TrackedResource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>,
    /// `DECANT_D3D12X_PRESENT_PROBE=target`: a readback buffer for the most drawn other target.
    target_probe: Option<(crate::mem_resource::TrackedResource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RESOURCE_DESC)>,
    /// `DECANT_D3D12X_PRESENT_PROBE=fill`: one render target view slot for the marker square.
    marker: Option<ID3D12DescriptorHeap>,
    marker_charge: Option<crate::mem_owners::Charge>,
    frames: u64,
}

impl Drop for Presenter {
    fn drop(&mut self) {
        if let Some(charge) = self.marker_charge {
            crate::mem_owners::remove_charge(charge);
        }
    }
}

/// Whether `DECANT_D3D12X_PRESENT_PROBE=fill` asks for a magenta square on every shown frame, which
/// proves the path from the swap chain to the screen whatever the title drew.
fn marker_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DECANT_D3D12X_PRESENT_PROBE").is_some_and(|v| v == "fill"))
}

/// What a readback of the presented resource holds: the pixels looked at, how many are not zero
/// in their colour channels, how many are not zero at all, the pixel at the centre, and the box
/// (left, top, right, bottom) around the coloured pixels.
#[derive(Debug, PartialEq, Eq)]
pub struct Probe {
    pub sampled: u64,
    pub colour: u64,
    pub any: u64,
    pub centre: u32,
    pub bbox: Option<(u32, u32, u32, u32)>,
}

/// The bytes per pixel of a format the probe reads (32 bit formats only), `None` for the others.
fn probe_size(f: DXGI_FORMAT) -> Option<u32> {
    matches!(f.0, 24..=29 | 39..=45 | 87..=93).then_some(4)
}

/// Looks at every pixel of a 32 bit per pixel image of `w` by `h` with
/// rows `pitch` bytes apart. Colour is the low 24 bits (R8G8B8A8 and B8G8R8A8 alike, the
/// alpha is the top byte).
pub fn probe_pixels(bytes: &[u8], w: u32, h: u32, pitch: usize) -> Probe {
    let px = |x: u32, y: u32| -> u32 {
        let at = y as usize * pitch + x as usize * 4;
        bytes.get(at..at + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let mut p = Probe { sampled: 0, colour: 0, any: 0, centre: px(w / 2, h / 2), bbox: None };
    for y in 0..h {
        for x in 0..w {
            let v = px(x, y);
            p.sampled += 1;
            if v & 0x00FF_FFFF != 0 {
                p.colour += 1;
                let b = p.bbox.get_or_insert((x, y, x, y));
                *b = (b.0.min(x), b.1.min(y), b.2.max(x), b.3.max(y));
            }
            p.any += u64::from(v != 0);
        }
    }
    p
}

fn probe_enabled() -> bool {
    crate::rt_diag::enabled()
}

/// Whether `DECANT_D3D12X_PRESENT_PROBE=target` also reads back the render target with the most draws
/// besides the presented one. Its state is not tracked: it is taken to be a shader resource (the
/// title sampled it for the frame), which is a guess, so this is a diagnosis only.
fn target_probe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DECANT_D3D12X_PRESENT_PROBE").is_some_and(|v| v == "target"))
}

/// The size, format, mip count and flags of PC resource `r`, for the probe line.
fn describe(r: *mut c_void) -> String {
    // SAFETY: the render target map holds PC resources the title made views for, and the probe runs
    // while they are in use, so they are alive.
    let Some(res) = (unsafe { ID3D12Resource::from_raw_borrowed(&r) }) else { return "none".to_owned() };
    // SAFETY: a plain query on a live resource.
    let d = unsafe { res.GetDesc() };
    format!("{}x{} format {} mips {} flags {:#x}", d.Width, d.Height, d.Format.0, d.MipLevels, d.Flags.0)
}

/// Makes a readback buffer that holds subresource 0 of a resource of description `d`.
fn make_probe(dev: &ID3D12Device, d: &D3D12_RESOURCE_DESC) -> windows::core::Result<(crate::mem_resource::TrackedResource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0u64;
    // SAFETY: COM calls on a live device with valid descriptions and out pointers.
    unsafe {
        dev.GetCopyableFootprints(d, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let buf = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut out)?;
        let resource = out.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))?;
        Ok((crate::mem_resource::TrackedResource::readback(resource, total), fp))
    }
}

/// `DECANT_D3D12X_PROBE_DUMP=<dir>`: where to write pictures of the probe's readbacks, read once.
fn dump_dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("DECANT_D3D12X_PROBE_DUMP").map(PathBuf::from)).as_deref()
}

static DUMP_SAMPLES: AtomicU64 = AtomicU64::new(0);
static DUMP_FILES: AtomicU64 = AtomicU64::new(0);
/// Formats already logged as "not decoded, dump skipped" (logged once per format, not once per frame).
static DUMP_FORMAT_WARNED: Mutex<Vec<i32>> = Mutex::new(Vec::new());

/// Parses `DECANT_D3D12X_PROBE_BURST=<first frame>,<count>`.
fn parse_probe_burst(value: &str) -> Option<(u64, u64)> {
    let (first, count) = value.trim().split_once(',')?;
    if count.contains(',') {
        return None;
    }
    let first = first.trim().parse::<u64>().ok()?;
    let count = count.trim().parse::<u64>().ok()?;
    (count > 0 && first.checked_add(count - 1).is_some()).then_some((first, count))
}

fn probe_burst() -> Option<(u64, u64)> {
    static BURST: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    BURST.get_or_init(|| std::env::var("DECANT_D3D12X_PROBE_BURST").ok().and_then(|v| parse_probe_burst(&v))).to_owned()
}

fn burst_frame(frame: u64) -> bool {
    probe_burst().is_some_and(|(first, count)| frame >= first && frame - first < count)
}

fn dump_budget_allows(files: u64) -> bool {
    files < 40
}

fn burst_dump_due(frame: u64, burst: (u64, u64), files: u64) -> bool {
    dump_budget_allows(files) && frame >= burst.0 && frame - burst.0 < burst.1
}

/// Whether this probe sample should be dumped: at most once every 8 samples (`DECANT_D3D12X_PROBE_DUMP_EVERY` sets another number), and only while
/// under the 40 files per run budget (4K frames are tens of MB each). Advances the sample
/// counter regardless, so the cadence is based on samples taken, not files written.
fn dump_due(frame: u64) -> bool {
    let n = DUMP_SAMPLES.fetch_add(1, Ordering::Relaxed);
    let files = DUMP_FILES.load(Ordering::Relaxed);
    if let Some((first, count)) = probe_burst() {
        return burst_dump_due(frame, (first, count), files);
    }
    if !dump_budget_allows(files) {
        return false;
    }
    static EVERY: OnceLock<u64> = OnceLock::new();
    let every = *EVERY.get_or_init(|| std::env::var("DECANT_D3D12X_PROBE_DUMP_EVERY").ok().and_then(|v| v.trim().parse().ok()).filter(|n| *n > 0).unwrap_or(8));
    n.is_multiple_of(every) && DUMP_FILES.load(Ordering::Relaxed) < 40
}

/// Where the 4 byte texel at pixel `(x, y)` starts inside a mapped buffer of `buf_bytes` bytes
/// that is laid out by footprint `fp`. `None` when the texel would stick out of its row
/// (`RowPitch` narrower than 4 bytes per pixel) or out of the buffer, or when an offset overflows.
fn readback_texel_offset(fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, buf_bytes: usize, x: usize, y: usize) -> Option<usize> {
    let pitch = fp.Footprint.RowPitch as usize;
    let column = x.checked_mul(4)?;
    if column.checked_add(4)? > pitch {
        return None;
    }
    let start = usize::try_from(fp.Offset).ok()?;
    let at = y.checked_mul(pitch)?.checked_add(column)?.checked_add(start)?;
    (at.checked_add(4)? <= buf_bytes).then_some(at)
}

/// One texel as 8 bit RGB, for the two formats the dump decodes: `R8G8B8A8_UNORM` (28) and the
/// 10 bit back buffer format `R10G10B10A2_UNORM`. Every other format, the typeless, sRGB and BGRA
/// siblings included, gives `None`, so the caller skips the dump instead of writing a picture with
/// wrong colours.
fn dump_rgb(format: DXGI_FORMAT, texel: u32) -> Option<[u8; 3]> {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => {
            let [r, g, b, _] = texel.to_le_bytes();
            Some([r, g, b])
        }
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let top8 = |shift: u32| ((texel >> shift) & 0x3FF) as u16 >> 2;
            Some([top8(0) as u8, top8(10) as u8, top8(20) as u8])
        }
        _ => None,
    }
}

/// Size of the BMP file header plus the info header.
const BMP_HEADER_BYTES: usize = 54;

/// The BMP headers of an uncompressed 24 bit image of `w` by `h` pixels and `file_bytes` in all.
fn bmp_header(w: u32, h: u32, file_bytes: u32) -> [u8; BMP_HEADER_BYTES] {
    let mut head = [0u8; BMP_HEADER_BYTES];
    head[0..2].copy_from_slice(b"BM");
    head[2..6].copy_from_slice(&file_bytes.to_le_bytes());
    head[10..14].copy_from_slice(&(BMP_HEADER_BYTES as u32).to_le_bytes());
    head[14..18].copy_from_slice(&40u32.to_le_bytes());
    head[18..22].copy_from_slice(&w.to_le_bytes());
    head[22..26].copy_from_slice(&h.to_le_bytes());
    head[26..28].copy_from_slice(&1u16.to_le_bytes());
    head[28..30].copy_from_slice(&24u16.to_le_bytes());
    head
}

/// Writes `bytes`, a mapped readback buffer laid out by `fp`, to `path` as an uncompressed 24 bit
/// BMP at full resolution. Rows run bottom up and are padded to 4 bytes. A texel that cannot be
/// read, or whose format is not decoded, comes out black.
fn write_bmp(path: &Path, bytes: &[u8], fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, format: DXGI_FORMAT) -> std::io::Result<()> {
    let (w, h) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize);
    let padded_row = (w * 3).next_multiple_of(4);
    let total = BMP_HEADER_BYTES + padded_row * h;
    let mut file = Vec::with_capacity(total);
    file.extend_from_slice(&bmp_header(w as u32, h as u32, total as u32));
    for y in (0..h).rev() {
        let row_start = file.len();
        for x in 0..w {
            let rgb = readback_texel_offset(fp, bytes.len(), x, y)
                .and_then(|at| dump_rgb(format, u32::from_le_bytes(bytes[at..at + 4].try_into().ok()?)))
                .unwrap_or_default();
            file.extend_from_slice(&[rgb[2], rgb[1], rgb[0]]);
        }
        file.resize(row_start + padded_row, 0);
    }
    std::fs::write(path, &file)
}

/// Writes `tag`'s readback (`bytes` mapped under footprint `fp`, pixel format `format`) as
/// `<dir>/frame<frame>-<tag>.bmp`, counted against the 40 files per run budget. Logs the result;
/// an undecoded format is logged once (not once per frame) and nothing is written.
fn dump_readback(dir: &Path, frame: u64, tag: &str, bytes: &[u8], fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, format: DXGI_FORMAT) {
    if !dump_budget_allows(DUMP_FILES.load(Ordering::Relaxed)) {
        return;
    }
    if dump_rgb(format, 0).is_none() {
        let mut warned = DUMP_FORMAT_WARNED.lock().unwrap_or_else(|e| e.into_inner());
        if !warned.contains(&format.0) {
            warned.push(format.0);
            xs_log!("present probe: dump: format {} is not decoded, {tag} dump skipped", format.0);
        }
        return;
    }
    let path = dir.join(format!("frame{frame}-{tag}.bmp"));
    match write_bmp(&path, bytes, fp, format) {
        Ok(()) => {
            DUMP_FILES.fetch_add(1, Ordering::Relaxed);
            xs_log!("present probe: dump: wrote {}x{} to {}", fp.Footprint.Width, fp.Footprint.Height, path.display());
        }
        Err(e) => xs_log!("present probe: dump: could not write {}: {e}", path.display()),
    }
}

/// Writes the present probe facts that accompany a burst image.
fn dump_burst_info(dir: &Path, frame: u64, resource: *mut c_void) {
    if !dump_budget_allows(DUMP_FILES.load(Ordering::Relaxed)) {
        return;
    }
    let path = dir.join(format!("frame{frame}-info.txt"));
    let body = format!("frame {frame}\npresented resource {resource:p}\n{}\nconstant buffers: {}\n", crate::rt_diag::summary(resource as usize), crate::view_diag::watched_now());
    match std::fs::write(&path, body) {
        Ok(()) => {
            DUMP_FILES.fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => xs_log!("present probe: dump: could not write {}: {e}", path.display()),
    }
}

// SAFETY: the D3D12 and DXGI objects are free threaded; the mutex below serialises their use.
unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

/// The largest visible top level window of this process: the title's.
fn title_window() -> Option<HWND> {
    struct Found {
        pid: u32,
        best: Option<(HWND, i64)>,
    }
    unsafe extern "system" fn each(hwnd: HWND, data: LPARAM) -> BOOL {
        // SAFETY: `data` is the `Found` passed to `EnumWindows` below, alive for the call.
        let found = unsafe { &mut *(data.0 as *mut Found) };
        let mut pid = 0u32;
        // SAFETY: plain queries on a window handle the system handed over.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
            if pid == found.pid && IsWindowVisible(hwnd).as_bool() {
                let mut r = RECT::default();
                if GetClientRect(hwnd, &mut r).is_ok() {
                    let area = i64::from(r.right - r.left) * i64::from(r.bottom - r.top);
                    if found.best.is_none_or(|(_, a)| area > a) {
                        found.best = Some((hwnd, area));
                    }
                }
            }
        }
        BOOL(1)
    }
    // SAFETY: a plain query.
    let mut found = Found { pid: unsafe { GetCurrentProcessId() }, best: None };
    // SAFETY: the callback only uses `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(each), LPARAM(&mut found as *mut Found as isize)) };
    found.best.map(|(h, _)| h)
}

/// The client size for a window in a work area of `work` (width, height) whose frame adds `frame`
/// (width, height): the largest 16:9 client that makes the whole window fit 90 % of the work area,
/// never larger than the `back` buffer.
fn fit_client(work: (i32, i32), frame: (i32, i32), back: (u32, u32)) -> (i32, i32) {
    let avail_w = (work.0 * 9 / 10 - frame.0).max(160);
    let avail_h = (work.1 * 9 / 10 - frame.1).max(90);
    let mut w = avail_w.min(avail_h * 16 / 9).min(back.0 as i32).min(back.1 as i32 * 16 / 9);
    w &= !1;
    (w, w * 9 / 16)
}

/// The top left corner that centres an `outer` sized window in the work area `(left, top, width, height)`.
fn centre(work: (i32, i32, i32, i32), outer: (i32, i32)) -> (i32, i32) {
    (work.0 + (work.2 - outer.0) / 2, work.1 + (work.3 - outer.1) / 2)
}

/// Parses `<width>x<height>` of DECANT_D3D12X_WINDOW_SIZE.
fn parse_size(s: &str) -> Option<(i32, i32)> {
    let (w, h) = s.trim().split_once(['x', 'X'])?;
    let (w, h) = (w.trim().parse::<i32>().ok()?, h.trim().parse::<i32>().ok()?);
    (w >= 160 && h >= 90 && w <= 16384 && h <= 16384).then_some((w, h))
}

/// Makes the title's window a normal one: overlapped style, a 16:9 client that fits the work area of
/// its monitor, centred. Once, when the layer first learns the window. Does not follow later changes.
fn fit_window(hwnd: HWND, back: (u32, u32)) {
    // SAFETY: plain window and monitor queries and changes on a window of this process.
    unsafe {
        let mut style = GetWindowLongPtrW(hwnd, GWL_STYLE) as u32;
        if style & WS_OVERLAPPEDWINDOW.0 != WS_OVERLAPPEDWINDOW.0 {
            style = (style & !WS_POPUP.0) | WS_OVERLAPPEDWINDOW.0;
            SetWindowLongPtrW(hwnd, GWL_STYLE, style as isize);
        }
        if IsZoomed(hwnd).as_bool() {
            let _ = ShowWindow(hwnd, SW_RESTORE);
        }
        let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32;
        let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
        if !GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mut mi).as_bool() {
            xs_log!("present: window fit skipped, no monitor info");
            return;
        }
        let wa = mi.rcWork;
        let work = (wa.right - wa.left, wa.bottom - wa.top);
        let mut f = RECT::default();
        let _ = AdjustWindowRectEx(&mut f, WINDOW_STYLE(style), false, WINDOW_EX_STYLE(ex));
        let frame = (f.right - f.left, f.bottom - f.top);
        let over = std::env::var("DECANT_D3D12X_WINDOW_SIZE").ok().and_then(|s| parse_size(&s));
        let (cw, ch) = over.unwrap_or_else(|| fit_client(work, frame, back));
        let (ow, oh) = (cw + frame.0, ch + frame.1);
        let (x, y) = centre((wa.left, wa.top, work.0, work.1), (ow, oh));
        let _ = SetWindowPos(hwnd, None, x, y, ow, oh, SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW);
        let (mut c, mut w) = (RECT::default(), RECT::default());
        let _ = GetClientRect(hwnd, &mut c);
        let _ = GetWindowRect(hwnd, &mut w);
        xs_log!(
            "present: window fit: work area {}x{} at {},{}, client {}x{} asked, got {}x{}, window rect {},{} to {},{}",
            work.0, work.1, wa.left, wa.top, cw, ch, c.right - c.left, c.bottom - c.top, w.left, w.top, w.right, w.bottom
        );
    }
}

fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed copy of the pointer, no reference taken; the barrier does not
                // outlive `resource` and is never dropped as an owner.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Makes the presenter for `queue`, with a swap chain on the title's window for a `w` by `h`
/// back buffer of format `fmt`.
fn make(queue: &ID3D12CommandQueue, w: u32, h: u32, fmt: DXGI_FORMAT) -> windows::core::Result<Presenter> {
    let hwnd = title_window().ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))?;
    // The title reads its window messages rarely (its main thread builds pipelines for seconds at
    // a time), and after five seconds Windows covers a window that does not answer with a white
    // ghost copy, which hides the frames presented here. No ghost for this process.
    // SAFETY: a process wide switch without arguments.
    unsafe { windows::Win32::UI::WindowsAndMessaging::DisableProcessWindowsGhosting() };
    // The title's thread may not pump messages for seconds, and changing a window of another thread
    // waits for it. So the fit runs on its own thread and never holds up the presenting one.
    let raw = hwnd.0 as isize;
    let _ = std::thread::Builder::new().name("window-fit".into()).spawn(move || fit_window(HWND(raw as *mut c_void), (w, h)));
    // SAFETY: COM calls on live objects with valid descriptions.
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        queue.GetDevice(&mut dev)?;
        let dev = dev.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))?;
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: w,
            Height: h,
            Format: fmt,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            ..Default::default()
        };
        let swap: IDXGISwapChain3 = factory.CreateSwapChainForHwnd(queue, hwnd, &desc, None, None)?.cast()?;
        let _ = factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
        let allocators = [dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?, dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?];
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocators[0], None)?;
        list.Close()?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        xs_log!("present: swap chain {w}x{h} format {} on the title window {:?}", fmt.0, hwnd.0);
        Ok(Presenter { queue: queue.clone(), swap, size: (w, h, fmt), allocators, list, fence, marks: [0; 2], next: 0, probe: None, target_probe: None, marker: None, marker_charge: None, frames: 0 })
    }
}

/// Flush recorded trace events periodically so long runs keep trace JSON on disk.
const TRACE_FLUSH_INTERVAL: u64 = 300;

/// Copies `plane0` into the next back buffer, optionally composes `plane1` over it with
/// premultiplied alpha (`present_over`, `DECANT_D3D12X_PRESENT_OVER`), and presents. If `plane0` is
/// null and `plane1` is non-null, `plane1` is shown directly. False when it cannot be shown
/// (not enabled, no window, a format a swap chain does not take, a failed call), logged once.
///
/// # Safety
/// `queue` is a live PC `ID3D12CommandQueue`; each non-null plane is a live PC `ID3D12Resource`.
pub unsafe fn show_planes(queue: *mut c_void, plane0: *mut c_void, plane1: *mut c_void) -> bool {
    static FAILED: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
    let fail = |why: String| {
        if FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            xs_log!("present: frame not shown: {why}");
        }
        false
    };
    let (resource, overlay) = if plane0.is_null() { (plane1, std::ptr::null_mut()) } else { (plane0, plane1) };
    if !enabled() || queue.is_null() || resource.is_null() {
        return false;
    }
    let _trace = decant_trace::span("present", "show");
    let overlay = if !overlay.is_null() && overlay != resource && crate::present_over::enabled() { overlay } else { std::ptr::null_mut() };
    // SAFETY: by the contract above.
    let (Some(queue), Some(source)) = (unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }, unsafe { ID3D12Resource::from_raw_borrowed(&resource) }) else {
        return false;
    };
    // SAFETY: by the contract above; null yields None.
    let over_res = unsafe { ID3D12Resource::from_raw_borrowed(&overlay) };
    // SAFETY: a plain query on a live resource.
    let d = unsafe { source.GetDesc() };
    let Some(fmt) = swap_format(d.Format) else {
        return fail(format!("back buffer format {} has no swap chain form", d.Format.0));
    };
    let (w, h) = (d.Width as u32, d.Height);
    let mut guard = decant_trace::timed_lock("present::PRESENTER", || PRESENTER.lock().unwrap_or_else(|e| e.into_inner()));
    if guard.as_ref().is_none_or(|p| p.size != (w, h, fmt) || p.queue.as_raw() != queue.as_raw()) {
        *guard = None;
        match make(queue, w, h, fmt) {
            Ok(p) => *guard = Some(p),
            Err(e) => return fail(format!("no swap chain: {e}")),
        }
    }
    let Some(p) = guard.as_mut() else { return false };
    // SAFETY: COM calls on the presenter's live objects; each back buffer is reused only after its
    // last copy completed on the GPU.
    let result: windows::core::Result<u64> = unsafe {
        (|| {
            let i = p.swap.GetCurrentBackBufferIndex() as usize % 2;
            if p.fence.GetCompletedValue() < p.marks[i] {
                let _wait = decant_trace::span("present", "fence_wait");
                // A null event makes the call wait until the value is reached.
                p.fence.SetEventOnCompletion(p.marks[i], windows::Win32::Foundation::HANDLE::default())?;
            }
            p.allocators[i].Reset()?;
            p.list.Reset(&p.allocators[i], None)?;
            let back: ID3D12Resource = p.swap.GetBuffer(i as u32)?;
            // A console title usually presents its back buffer in RENDER_TARGET, but it may also
            // leave it in COMMON or PRESENT. The tracked state avoids a mismatch with the barrier.
            let src_state = source_state(source.as_raw() as usize);
            let mut before_len = 0usize;
            let mut before = [D3D12_RESOURCE_BARRIER::default(), D3D12_RESOURCE_BARRIER::default()];
            if src_state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                before[before_len] = transition(source, src_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                before_len += 1;
            }
            before[before_len] = transition(&back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
            before_len += 1;
            p.list.ResourceBarrier(&before[..before_len]);
            p.list.CopyResource(&back, source);
            p.frames += 1;
            let probing = probe_enabled() && (p.frames <= 4 || p.frames % 16 == 0 || (dump_dir().is_some() && burst_frame(p.frames)));
            if probing && p.probe.is_none() {
                let mut dev: Option<ID3D12Device> = None;
                p.queue.GetDevice(&mut dev)?;
                if let Some(dev) = dev {
                    p.probe = make_probe(&dev, &d).ok();
                }
            }
            let probe = if probing { p.probe.as_ref().map(|(b, f)| (b.resource.clone(), *f)) } else { None };
            if let Some((buf, fp)) = &probe {
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(buf),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(source),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                p.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            // The most drawn other target this frame, read back next to the presented one.
            // `most_drawn_other` hands back its own owned reference (`rt_diag::OwnedTarget`):
            // the resource named is genuinely alive, never a revived dead pointer, and stays
            // alive exactly as long as `other` is in scope (through the fence wait below).
            let other = if probe.is_some() && target_probe_enabled() { crate::rt_diag::most_drawn_other(resource as usize) } else { None };
            // This frame's draws are read; reset the bookkeeping for the next one. Independent
            // of `other`, which keeps its own reference regardless of what happens here.
            crate::rt_diag::end_frame();
            let mut target_read = None;
            if let Some(owned) = &other {
                let raw = owned.raw() as *mut c_void;
                // SAFETY: `owned` holds its own live COM reference, so this never borrows a
                // dangling pointer. `from_raw_borrowed` returning `None` here would mean `raw`
                // is null, which `OwnedTarget` never wraps; skip the target read rather than
                // abort the whole present on a defensive check that should never trip.
                if let Some(o) = ID3D12Resource::from_raw_borrowed(&raw) {
                    let od = o.GetDesc();
                    if p.target_probe.as_ref().is_none_or(|t| t.2.Width != od.Width || t.2.Height != od.Height || t.2.Format != od.Format) {
                        let mut dev: Option<ID3D12Device> = None;
                        p.queue.GetDevice(&mut dev)?;
                        p.target_probe = dev.and_then(|dev| make_probe(&dev, &od).ok()).map(|(b, f)| (b, f, od));
                    }
                    if let Some((buf, fp, od)) = p.target_probe.as_ref().map(|(b, f, d)| (b.resource.clone(), *f, *d))
                        && od.Format.0 != 0
                        && probe_size(od.Format).is_some()
                    {
                        let sr = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                        p.list.ResourceBarrier(&[transition(o, sr, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&buf), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(o), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                        p.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                        p.list.ResourceBarrier(&[transition(o, D3D12_RESOURCE_STATE_COPY_SOURCE, sr)]);
                        target_read = Some((buf, fp, od, o.as_raw()));
                    }
                }
            }
            let mut back_state = D3D12_RESOURCE_STATE_COPY_DEST;
            if let Some(over) = over_res {
                let od = over.GetDesc();
                if let Some(ofmt) = swap_format(od.Format) {
                    let mut dev: Option<ID3D12Device> = None;
                    p.queue.GetDevice(&mut dev)?;
                    if let Some(dev) = dev {
                        p.list.ResourceBarrier(&[
                            transition(&back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET),
                            transition(over, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                        ]);
                        if crate::present_over::record(&dev, &p.list, &back, fmt, (w, h), over, ofmt) {
                            static COMPOSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                            if !COMPOSED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                xs_log!("present: second plane composed over plane 0 ({}x{} format {} over {w}x{h} format {})", od.Width, od.Height, od.Format.0, d.Format.0);
                            }
                        }
                        p.list.ResourceBarrier(&[transition(over, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
                        back_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
                    }
                }
            }
            if marker_enabled() {
                if p.marker.is_none() {
                    let mut dev: Option<ID3D12Device> = None;
                    p.queue.GetDevice(&mut dev)?;
                    if let Some(dev) = dev {
                        let desc = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() };
                        p.marker = dev.CreateDescriptorHeap(&desc).ok();
                        if p.marker.is_some() {
                            let increment = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
                            let charge = crate::mem_owners::Charge {
                                kind: crate::mem_owners::Kind::DescriptorHeap,
                                segment: crate::mem_owners::Segment::Local,
                                bytes: u64::from(increment),
                            };
                            crate::mem_owners::add_charge(charge);
                            p.marker_charge = Some(charge);
                        }
                    }
                }
                if let Some(heap) = &p.marker {
                    let mut dev: Option<ID3D12Device> = None;
                    p.queue.GetDevice(&mut dev)?;
                    let rtv = heap.GetCPUDescriptorHandleForHeapStart();
                    if let Some(dev) = dev {
                        dev.CreateRenderTargetView(&back, None, rtv);
                    }
                    if back_state != D3D12_RESOURCE_STATE_RENDER_TARGET {
                        p.list.ResourceBarrier(&[transition(&back, back_state, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
                        back_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
                    }
                    let square = RECT { left: 64, top: 64, right: 576, bottom: 576 };
                    p.list.ClearRenderTargetView(rtv, &[1.0, 0.0, 1.0, 1.0], Some(&[square]));
                }
            }
            let mut after_len = 0usize;
            let mut after = [D3D12_RESOURCE_BARRIER::default(), D3D12_RESOURCE_BARRIER::default()];
            if src_state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                after[after_len] = transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE, src_state);
                after_len += 1;
            }
            after[after_len] = transition(&back, back_state, D3D12_RESOURCE_STATE_PRESENT);
            after_len += 1;
            p.list.ResourceBarrier(&after[..after_len]);
            p.list.Close()?;
            // the copy reads the title's target: a release of it waits for this submission too
            let list: ID3D12CommandList = p.list.cast()?;
            let tracked = crate::retire::begin(p.queue.as_raw());
            p.queue.ExecuteCommandLists(&[Some(list)]);
            if tracked {
                crate::retire::submitted(p.queue.as_raw());
            }
            {
                let _dxgi = decant_trace::span("present", "dxgi_present");
                p.swap.Present(0, DXGI_PRESENT(0)).ok()?;
            }
            p.next += 1;
            p.queue.Signal(&p.fence, p.next)?;
            p.marks[i] = p.next;
            if let Some((buf, fp)) = &probe {
                // Diagnostics only: wait for the copy, then read the pixels.
                p.fence.SetEventOnCompletion(p.next, windows::Win32::Foundation::HANDLE::default())?;
                let mut at: *mut c_void = std::ptr::null_mut();
                buf.Map(0, None, Some(&mut at))?;
                let pitch = fp.Footprint.RowPitch as usize;
                let len = pitch * fp.Footprint.Height as usize;
                let bytes = std::slice::from_raw_parts(at.cast::<u8>(), len);
                let r = probe_pixels(bytes, w, h, pitch);
                // One cadence decision per sample, shared by the presented and target dumps
                // below (a "pair"), so a `target` mode run does not dump twice as often.
                let do_dump = dump_dir().is_some() && dump_due(p.frames);
                if do_dump
                    && let Some(dir) = dump_dir()
                {
                    dump_readback(dir, p.frames, "presented", bytes, fp, d.Format);
                    if burst_frame(p.frames) {
                        dump_burst_info(dir, p.frames, resource);
                    }
                }
                buf.Unmap(0, None);
                xs_log!(
                    "present probe: frame {} (thread {}), resource {:p}: {} of {} pixels have colour (box {:?}), {} are not zero, centre {:08X}; {}",
                    p.frames,
                    windows_sys::Win32::System::Threading::GetCurrentThreadId(),
                    resource,
                    r.colour,
                    r.sampled,
                    r.bbox,
                    r.any,
                    r.centre,
                    crate::rt_diag::summary(resource as usize)
                );
                if crate::rt_diag::enabled() {
                    xs_log!("present probe: constant buffers of described draws now: {}", crate::view_diag::watched_now());
                }
                if let Some((buf, fp, od, raw)) = &target_read {
                    let mut at: *mut c_void = std::ptr::null_mut();
                    buf.Map(0, None, Some(&mut at))?;
                    let pitch = fp.Footprint.RowPitch as usize;
                    let bytes = std::slice::from_raw_parts(at.cast::<u8>(), pitch * fp.Footprint.Height as usize);
                    let r = probe_pixels(bytes, od.Width as u32, od.Height, pitch);
                    if do_dump
                        && let Some(dir) = dump_dir()
                    {
                        dump_readback(dir, p.frames, "target", bytes, fp, od.Format);
                    }
                    buf.Unmap(0, None);
                    xs_log!(
                        "present probe: frame {}, most drawn other target {:p} ({}): {} of {} pixels have colour (box {:?}), {} are not zero, centre {:08X}",
                        p.frames,
                        *raw,
                        describe(*raw),
                        r.colour,
                        r.sampled,
                        r.bbox,
                        r.any,
                        r.centre
                    );
                }
            }
            Ok(p.frames)
        })()
    };
    drop(guard);
    match result {
        Ok(frame_num) => {
            static SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SHOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                xs_log!("present: first frame shown ({w}x{h}, format {})", d.Format.0);
            }
            if decant_trace::enabled() {
                decant_trace::frame_mark();
                if frame_num.is_multiple_of(TRACE_FLUSH_INTERVAL) {
                    let _ = decant_trace::flush();
                }
            }
            true
        }
        Err(e) => {
            *PRESENTER.lock().unwrap_or_else(|e| e.into_inner()) = None;
            fail(format!("{e}"))
        }
    }
}

#[derive(Default)]
struct StateTracker {
    resources: HashMap<usize, u32>,
    list_notes: HashMap<usize, Vec<(usize, u32)>>,
}

static STATES: Mutex<Option<StateTracker>> = Mutex::new(None);

fn with_states<R>(f: impl FnOnce(&mut StateTracker) -> R) -> R {
    let mut guard = STATES.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(StateTracker::default))
}

/// Records the initial `D3D12_RESOURCE_STATES` of a newly created PC resource.
pub(crate) fn note_resource_created(res: usize, state: u32) {
    if res != 0 {
        with_states(|s| {
            s.resources.insert(res, state);
        });
    }
}

/// Records transition barriers recorded on PC command list `list` for subresource 0 / all subresources.
pub(crate) fn note_list_transitions(list: usize, transitions: &[(usize, u32, u32)]) {
    if list == 0 || transitions.is_empty() {
        return;
    }
    let kept: Vec<(usize, u32)> = transitions
        .iter()
        .filter(|(res, sub, _)| *res != 0 && (*sub == 0 || *sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES))
        .map(|&(res, _, after)| (res, after))
        .collect();
    if !kept.is_empty() {
        with_states(|s| {
            s.list_notes.entry(list).or_default().extend(kept);
        });
    }
}

/// Clears recorded transitions when PC command list `list` is reset or released.
pub(crate) fn list_reset(list: usize) {
    if list != 0 {
        with_states(|s| {
            if let Some(entries) = s.list_notes.get_mut(&list) {
                entries.clear();
            }
        });
    }
}

/// Applies recorded transitions of submitted PC command lists in execution order.
pub(crate) fn list_submitted(lists: &[usize]) {
    if lists.is_empty() {
        return;
    }
    with_states(|s| {
        let StateTracker { resources, list_notes } = s;
        for &list in lists {
            if let Some(notes) = list_notes.get(&list) {
                for &(res, after) in notes {
                    resources.insert(res, after);
                }
            }
        }
    });
}

/// Drops tracked state for a released PC resource.
pub(crate) fn forget_resource(res: usize) {
    if res != 0 {
        with_states(|s| {
            s.resources.remove(&res);
        });
    }
}

/// The tracked state of `res`, `RENDER_TARGET` when nothing is tracked. A title that created the
/// resource in or moved it to `COMMON` (0, the same value as `PRESENT`) before `PresentX` is
/// reported as such.
fn source_state(res: usize) -> D3D12_RESOURCE_STATES {
    with_states(|s| s.resources.get(&res).copied())
        .map_or(D3D12_RESOURCE_STATE_RENDER_TARGET, |v| D3D12_RESOURCE_STATES(v as i32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_burst_parser_accepts_a_first_frame_and_count() {
        assert_eq!(parse_probe_burst("12,4"), Some((12, 4)));
        assert_eq!(parse_probe_burst(" 12 , 4 "), Some((12, 4)));
        assert_eq!(parse_probe_burst("0,1"), Some((0, 1)));
    }

    #[test]
    fn probe_burst_parser_rejects_bad_or_empty_ranges() {
        for value in ["", "1", "1,", ",1", "1,0", "-1,2", "1,-2", "a,2", "1,b", "1,2,3", "18446744073709551615,2"] {
            assert_eq!(parse_probe_burst(value), None, "{value:?}");
        }
    }

    #[test]
    fn probe_burst_obeys_the_40_file_budget() {
        assert!(burst_dump_due(12, (12, 100), 39));
        assert!(!burst_dump_due(12, (12, 100), 40));
    }

    #[test]
    fn the_window_fits_the_work_area_in_16_by_9() {
        // 2560x1392 work area, frame 16x39: client at most 90 % high.
        let (w, h) = fit_client((2560, 1392), (16, 39), (3840, 2160));
        assert_eq!((w, h), (2156, 1212));
        assert!(h + 39 <= 1392 * 9 / 10 && w + 16 <= 2560);
        // never above the back buffer
        assert_eq!(fit_client((7680, 4320), (16, 39), (1920, 1080)), (1920, 1080));
        // tiny work area stays positive
        let (w, h) = fit_client((100, 50), (16, 39), (3840, 2160));
        assert!(w > 0 && h > 0);
    }

    #[test]
    fn the_window_is_centred_and_the_size_parsed() {
        assert_eq!(centre((0, 0, 2560, 1392), (2240, 1290)), (160, 51));
        assert_eq!(centre((-1920, 0, 1920, 1040), (1000, 800)), (-1460, 120));
        assert_eq!(parse_size("1600x900"), Some((1600, 900)));
        assert_eq!(parse_size(" 1280 X 720 "), Some((1280, 720)));
        assert_eq!(parse_size("0x0"), None);
        assert_eq!(parse_size("abc"), None);
    }

    #[test]
    fn the_presentable_formats_map_to_their_swap_chain_forms() {
        assert_eq!(swap_format(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB), Some(DXGI_FORMAT_R8G8B8A8_UNORM));
        assert_eq!(swap_format(DXGI_FORMAT_B8G8R8A8_TYPELESS), Some(DXGI_FORMAT_B8G8R8A8_UNORM));
        assert_eq!(swap_format(DXGI_FORMAT_R10G10B10A2_UNORM), Some(DXGI_FORMAT_R10G10B10A2_UNORM));
        assert_eq!(swap_format(DXGI_FORMAT_R11G11B10_FLOAT), None);
    }

    #[test]
    fn the_probe_counts_coloured_and_alpha_only_pixels() {
        let (w, h, pitch) = (32u32, 32u32, 32 * 4);
        let mut img = vec![0u8; pitch * h as usize];
        img[0..4].copy_from_slice(&0xFF00_0000u32.to_le_bytes());
        img[16 * 4..16 * 4 + 4].copy_from_slice(&0x0000_00FFu32.to_le_bytes());
        let c = 16 * pitch + 16 * 4;
        img[c..c + 4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        let r = probe_pixels(&img, w, h, pitch);
        assert_eq!(r, Probe { sampled: 32 * 32, colour: 2, any: 3, centre: 0x1234_5678, bbox: Some((16, 0, 16, 16)) });
    }

    fn footprint(width: u32, height: u32, pitch: u32, offset: u64) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: width, Height: height, Depth: 1, RowPitch: pitch },
        }
    }

    #[test]
    fn dump_rgb_decodes_exactly_two_formats() {
        assert_eq!(dump_rgb(DXGI_FORMAT_R8G8B8A8_UNORM, 0xAA33_2211), Some([0x11, 0x22, 0x33]));
        let ten_bit = (0x3FF << 20) | (0x200 << 10) | 0x100;
        assert_eq!(dump_rgb(DXGI_FORMAT_R10G10B10A2_UNORM, ten_bit), Some([0x40, 0x80, 0xFF]));
        for other in [DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_TYPELESS] {
            assert_eq!(dump_rgb(other, 0), None, "format {}", other.0);
        }
    }

    #[test]
    fn texel_offsets_follow_pitch_and_start_offset() {
        let fp = footprint(4, 4, 16, 8);
        assert_eq!(readback_texel_offset(&fp, 8 + 64, 0, 0), Some(8));
        assert_eq!(readback_texel_offset(&fp, 8 + 64, 2, 3), Some(8 + 3 * 16 + 8));
    }

    #[test]
    fn texels_outside_the_row_or_the_buffer_have_no_offset() {
        // A pitch of 12 holds three texels per row.
        let narrow = footprint(4, 2, 12, 0);
        assert_eq!(readback_texel_offset(&narrow, 24, 2, 0), Some(8));
        assert_eq!(readback_texel_offset(&narrow, 24, 3, 0), None);
        // The last row is cut off.
        let square = footprint(4, 4, 16, 0);
        assert_eq!(readback_texel_offset(&square, 48, 0, 3), None);
        assert_eq!(readback_texel_offset(&square, 51, 0, 3), None);
        assert_eq!(readback_texel_offset(&square, 52, 0, 3), Some(48));
        assert_eq!(readback_texel_offset(&square, 64, 0, 3), Some(48));
        // Overflows give None rather than wrapping.
        assert_eq!(readback_texel_offset(&footprint(4, 4, 16, u64::MAX), usize::MAX, 0, 0), None);
        assert_eq!(readback_texel_offset(&square, usize::MAX, usize::MAX, 0), None);
    }

    #[test]
    fn a_bmp_has_the_right_headers_and_bottom_up_bgr_rows() {
        let dir = std::env::temp_dir().join(format!("decant-d3d12x-bmp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.bmp");
        let bytes: [u8; 16] = [1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255];
        write_bmp(&path, &bytes, &footprint(2, 2, 8, 0), DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let out = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let word = |at: usize| u32::from_le_bytes(out[at..at + 4].try_into().unwrap());
        // A 2 pixel row is 6 bytes, padded to 8.
        assert_eq!(out.len(), 54 + 2 * 8);
        assert_eq!((&out[0..2], word(2), word(10), word(14)), (&b"BM"[..], 70, 54, 40));
        assert_eq!((word(18), word(22)), (2, 2));
        assert_eq!(u16::from_le_bytes([out[28], out[29]]), 24);
        // The bottom image row comes first, with blue first in each pixel.
        assert_eq!(&out[54..60], &[9, 8, 7, 12, 11, 10]);
        assert_eq!(&out[60..62], &[0, 0]);
        assert_eq!(&out[62..68], &[3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn an_undecodable_format_gives_a_black_bmp() {
        let dir = std::env::temp_dir().join(format!("decant-d3d12x-bmp-black-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.bmp");
        write_bmp(&path, &[200; 16], &footprint(2, 2, 8, 0), DXGI_FORMAT_B8G8R8A8_UNORM).unwrap();
        let out = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(out[54..].iter().all(|&b| b == 0));
    }

    #[test]
    fn dump_due_is_once_every_8_samples_until_the_file_budget_is_spent() {
        DUMP_SAMPLES.store(0, Ordering::Relaxed);
        DUMP_FILES.store(0, Ordering::Relaxed);
        assert!(dump_due(0), "sample 0 is due");
        for _ in 1..8 {
            assert!(!dump_due(0));
        }
        assert!(dump_due(0), "sample 8 is due again");
        DUMP_FILES.store(40, Ordering::Relaxed);
        DUMP_SAMPLES.store(120, Ordering::Relaxed);
        assert!(!burst_dump_due(0, (0, 100), DUMP_FILES.load(Ordering::Relaxed)), "the file budget also limits burst dumps");
        assert!(!dump_due(0), "the 40 files per run budget is spent");
    }

    #[test]
    fn tracks_source_state_from_common_to_render_target_and_back() {
        let res = 0xCAFE_0001usize;
        let list = 0xCAFE_0002usize;
        forget_resource(res);
        list_reset(list);

        // Untracked resource defaults to RENDER_TARGET.
        assert_eq!(source_state(res), D3D12_RESOURCE_STATE_RENDER_TARGET);

        // Created in COMMON -> tracked as COMMON.
        note_resource_created(res, D3D12_RESOURCE_STATE_COMMON.0 as u32);
        assert_eq!(source_state(res), D3D12_RESOURCE_STATE_COMMON);

        // Recorded transition to RENDER_TARGET does not take effect until ExecuteCommandLists.
        note_list_transitions(list, &[(res, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32)]);
        assert_eq!(source_state(res), D3D12_RESOURCE_STATE_COMMON);

        // After submission -> RENDER_TARGET.
        list_submitted(&[list]);
        assert_eq!(source_state(res), D3D12_RESOURCE_STATE_RENDER_TARGET);

        // Reset discards old transitions; new transition back to COMMON takes effect on submit.
        list_reset(list);
        note_list_transitions(list, &[(res, 0, D3D12_RESOURCE_STATE_COMMON.0 as u32)]);
        list_submitted(&[list]);
        assert_eq!(source_state(res), D3D12_RESOURCE_STATE_COMMON);

        forget_resource(res);
        list_reset(list);
    }

    #[test]
    fn list_reset_reuses_list_notes_allocation() {
        let res = 0xCAFE_0011usize;
        let list = 0xCAFE_0012usize;
        note_list_transitions(list, &[(res, 0, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32)]);
        let cap_before = with_states(|s| s.list_notes.get(&list).map(Vec::capacity));
        assert!(cap_before.is_some_and(|c| c >= 1));
        list_reset(list);
        let state_after = with_states(|s| s.list_notes.get(&list).map(|v| (v.len(), v.capacity())));
        assert_eq!(state_after, Some((0, cap_before.unwrap())));
        with_states(|s| {
            s.list_notes.remove(&list);
        });
    }
}
