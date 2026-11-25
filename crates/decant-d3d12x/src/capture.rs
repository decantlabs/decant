//! `DECANT_CAPTURE=<dir>`: pictures of what the title presents, as PNG files.
//!
//! * `DECANT_CAPTURE_FRAMES=<list>` picks the frames by number (the first present is frame 1):
//!   `1,2,10,600`, or `pow2` for 1, 2, 4, 8 and so on (the default).
//! * Each picked frame's presented resource is copied on the title's queue at `PresentX`, read back
//!   and written as `frame<n>-presented.png`.
//! * `DECANT_CAPTURE_TARGETS=1` also writes every render target drawn into during a picked frame, as
//!   `frame<n>-draw<k>-<w>x<h>.png`: the target 0 of a command list, taken after the last draw into
//!   it before the list changes state (a barrier, new render targets, or the end of the list),
//!   where it is still a render target. `k` is the number of the draw within the frame.
//!
//! Formats: 8 bit RGBA/BGRA (UNORM, SRGB, TYPELESS), 10 bit RGB, 16 bit and 32 bit float, R11G11B10
//! float, and one and two channel 8, 16 and 32 bit formats. Float formats are clamped to 0..1 and
//! sRGB encoded; the others are written as they are. Other formats are skipped and logged once.
//!
//! Off by default. Off, each hook is one relaxed atomic load.

#![allow(unsafe_code)]

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// Which presented frames are taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frames {
    /// Frames 1, 2, 4, 8 and so on.
    Pow2,
    /// Exactly these frame numbers, sorted and without repeats.
    List(Vec<u64>),
}

impl Frames {
    /// Reads `DECANT_CAPTURE_FRAMES`. Unset, blank or `pow2` mean the powers of two. Anything else
    /// must be a comma separated list of frame numbers.
    pub fn parse(value: Option<&str>) -> Result<Frames, String> {
        let text = value.map_or("", str::trim);
        if text.is_empty() || text.eq_ignore_ascii_case("pow2") {
            return Ok(Frames::Pow2);
        }
        let mut numbers = Vec::new();
        for piece in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            numbers.push(piece.parse::<u64>().map_err(|_| format!("{piece:?} is not a frame number"))?);
        }
        numbers.sort_unstable();
        numbers.dedup();
        Ok(Frames::List(numbers))
    }

    /// Whether frame number `frame` (the first present is 1) is taken.
    pub fn wants(&self, frame: u64) -> bool {
        match self {
            Frames::Pow2 => frame.is_power_of_two(),
            Frames::List(numbers) => numbers.binary_search(&frame).is_ok(),
        }
    }
}

struct Config {
    dir: PathBuf,
    frames: Frames,
    targets: bool,
}

/// The cached capture mode, see [`mode`]. 0 means not decided yet.
static STATE: AtomicU8 = AtomicU8::new(0);
static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
/// Frames presented so far. Draws recorded now belong to frame `PRESENTED + 1`.
static PRESENTED: AtomicU64 = AtomicU64::new(0);

/// A configuration a unit test installs, because a test cannot count on the environment.
#[cfg(test)]
static FORCED: OnceLock<Config> = OnceLock::new();

const MODE_OFF: u8 = 1;
const MODE_PRESENTED: u8 = 2;
const MODE_TARGETS: u8 = 3;

/// Reads the environment once.
fn read_config() -> Option<Config> {
    let dir = std::env::var_os("DECANT_CAPTURE").filter(|v| !v.is_empty()).map(PathBuf::from)?;
    let frames = Frames::parse(std::env::var("DECANT_CAPTURE_FRAMES").ok().as_deref()).unwrap_or_else(|e| {
        crate::log::write(format_args!("capture: DECANT_CAPTURE_FRAMES: {e}, taking the powers of two"));
        Frames::Pow2
    });
    let targets = std::env::var("DECANT_CAPTURE_TARGETS").is_ok_and(|v| !matches!(v.trim(), "" | "0"));
    let _ = std::fs::create_dir_all(&dir);
    let extra = if targets { ", with their render targets" } else { "" };
    crate::log::write(format_args!("capture: frames {frames:?} into {}{extra}", dir.display()));
    Some(Config { dir, frames, targets })
}

fn config() -> Option<&'static Config> {
    #[cfg(test)]
    if let Some(forced) = FORCED.get() {
        return Some(forced);
    }
    CONFIG.get_or_init(read_config).as_ref()
}

/// `MODE_OFF`, `MODE_PRESENTED` (presented frames only) or `MODE_TARGETS` (also the render targets).
fn mode() -> u8 {
    let known = STATE.load(Ordering::Relaxed);
    if known != 0 {
        return known;
    }
    let decided = match config() {
        None => MODE_OFF,
        Some(c) if c.targets => MODE_TARGETS,
        Some(_) => MODE_PRESENTED,
    };
    STATE.store(decided, Ordering::Relaxed);
    decided
}

/// Whether the frame being recorded right now is one whose render targets are taken.
fn targets_now() -> bool {
    mode() == MODE_TARGETS && config().is_some_and(|c| c.frames.wants(PRESENTED.load(Ordering::Relaxed) + 1))
}

// ---- pixel formats ----

/// An IEEE half float.
fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let fraction = i32::from(bits & 0x3FF);
    let magnitude = match exponent {
        31 => f32::INFINITY,
        0 => fraction as f32 * 2f32.powi(-24),
        e => (1024 + fraction) as f32 * 2f32.powi(e - 25),
    };
    sign * magnitude
}

/// An unsigned float with a 5 bit exponent and `man_bits` of mantissa, as in R11G11B10.
fn small_float(bits: u32, man_bits: u32) -> f32 {
    let exponent = ((bits >> man_bits) & 0x1F) as i32;
    let fraction = (bits & ((1 << man_bits) - 1)) as i32;
    let shift = man_bits as i32;
    match exponent {
        31 => f32::INFINITY,
        0 => fraction as f32 * 2f32.powi(-14 - shift),
        e => ((1 << shift) + fraction) as f32 * 2f32.powi(e - 15 - shift),
    }
}

/// A linear value clamped to 0..1 and encoded as 8 bit sRGB.
fn srgb(linear: f32) -> u8 {
    let v = if linear.is_nan() { 0.0 } else { linear.clamp(0.0, 1.0) };
    let encoded = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (encoded * 255.0 + 0.5) as u8
}

/// A value clamped to 0..1 as 8 bit, without a curve.
fn plain8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0) as u8
}

/// A `bits` wide unsigned normalised integer scaled to 8 bit, rounded to nearest.
fn unorm(v: u32, bits: u32) -> u8 {
    let max = (1u64 << bits) - 1;
    ((u64::from(v) * 255 + max / 2) / max) as u8
}

fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn lef32(b: &[u8], at: usize) -> f32 {
    f32::from_bits(le32(&b[at..]))
}

fn gray(v: u8) -> [u8; 4] {
    [v, v, v, 255]
}

/// Bytes per pixel and a decoder to RGBA8 for DXGI format number `format`, `None` for formats not
/// handled. The typeless variants are read as their usual typed sibling.
#[allow(clippy::type_complexity)]
pub fn decoder(format: i32) -> Option<(usize, fn(&[u8]) -> [u8; 4])> {
    Some(match format {
        // R32G32B32A32 typeless and float
        1 | 2 => (16, |b| [srgb(lef32(b, 0)), srgb(lef32(b, 4)), srgb(lef32(b, 8)), plain8(lef32(b, 12))]),
        // R16G16B16A16 typeless and float
        9 | 10 => (8, |b| [srgb(half(le16(b, 0))), srgb(half(le16(b, 2))), srgb(half(le16(b, 4))), plain8(half(le16(b, 6)))]),
        // R16G16B16A16_UNORM keeps the high byte of each channel
        11 => (8, |b| [b[1], b[3], b[5], b[7]]),
        // R10G10B10A2 typeless and unorm
        23 | 24 => (4, |b| {
            let v = le32(b);
            [unorm(v & 0x3FF, 10), unorm((v >> 10) & 0x3FF, 10), unorm((v >> 20) & 0x3FF, 10), unorm(v >> 30, 2)]
        }),
        // R11G11B10_FLOAT
        26 => (4, |b| {
            let v = le32(b);
            [srgb(small_float(v & 0x7FF, 6)), srgb(small_float((v >> 11) & 0x7FF, 6)), srgb(small_float(v >> 22, 5)), 255]
        }),
        // R8G8B8A8 typeless, unorm, unorm sRGB and uint
        27..=30 => (4, |b| [b[0], b[1], b[2], b[3]]),
        // R16G16_FLOAT
        34 => (4, |b| [srgb(half(le16(b, 0))), srgb(half(le16(b, 2))), 0, 255]),
        // R16G16_UNORM
        35 => (4, |b| [b[1], b[3], 0, 255]),
        // R32 typeless and float
        39 | 41 => (4, |b| gray(srgb(lef32(b, 0)))),
        // R8G8 typeless and unorm
        48 | 49 => (2, |b| [b[0], b[1], 0, 255]),
        // R16_FLOAT
        54 => (2, |b| gray(srgb(half(le16(b, 0))))),
        // R16_UNORM
        56 => (2, |b| gray(b[1])),
        // R8 typeless and unorm
        60 | 61 => (1, |b| gray(b[0])),
        // B8G8R8A8 unorm, typeless and sRGB
        87 | 90 | 91 => (4, |b| [b[2], b[1], b[0], b[3]]),
        // B8G8R8X8 unorm, typeless and sRGB
        88 | 92 | 93 => (4, |b| [b[2], b[1], b[0], 255]),
        _ => return None,
    })
}

/// The RGBA8 pixels of a `width` by `height` image of `format` whose rows are `pitch` bytes apart,
/// or `None` for an unhandled format or too few bytes.
pub fn decode(format: i32, bytes: &[u8], width: usize, height: usize, pitch: usize) -> Option<Vec<u8>> {
    let (size, to_rgba) = decoder(format)?;
    let row_len = width * size;
    let needed = if height == 0 { 0 } else { pitch * (height - 1) + row_len };
    if pitch < row_len || bytes.len() < needed {
        return None;
    }
    let mut rgba = Vec::with_capacity(width * height * 4);
    for row in 0..height {
        let start = row * pitch;
        for texel in bytes[start..start + row_len].chunks_exact(size) {
            rgba.extend_from_slice(&to_rgba(texel));
        }
    }
    Some(rgba)
}

/// Writes RGBA8 pixels to `path` as a PNG.
pub fn write_png(path: &std::path::Path, rgba: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    let sink = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = png::Encoder::new(sink, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(rgba).map_err(std::io::Error::other)?;
    writer.finish().map_err(std::io::Error::other)
}

#[cfg(windows)]
pub use platform::{bound, drew, flush, presented, rtv_made};

#[cfg(windows)]
mod platform {
    use super::*;
    use crate::log::xs_log;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::mem::ManuallyDrop;
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::Interface;

    /// The most bytes of render target readbacks kept for one frame.
    const FRAME_BUDGET: u64 = 2 << 30;
    /// The most render target pictures of one frame.
    const FRAME_TARGETS: usize = 400;

    struct Pending {
        frame: u64,
        draw: u64,
        buf: crate::mem_resource::TrackedResource,
        fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        desc: D3D12_RESOURCE_DESC,
    }

    #[derive(Default)]
    struct Tracking {
        /// Host RTV handle to the PC resource it was made for.
        rtv: HashMap<usize, usize>,
        /// Command list proxy to its render target 0.
        list_rt: HashMap<usize, usize>,
        /// Command list proxy to the target drawn into since its last copy, and the last draw's number.
        dirty: HashMap<usize, (usize, u64)>,
        /// The frame `draws` and `bytes` count for, and its draws and readback bytes so far.
        frame: u64,
        draws: u64,
        bytes: u64,
        taken: usize,
        pending: Vec<Pending>,
        warned: Vec<i32>,
    }

    static TRACK: Mutex<Option<Tracking>> = Mutex::new(None);

    fn with<R>(f: impl FnOnce(&mut Tracking) -> R) -> R {
        let mut g = TRACK.lock().unwrap_or_else(|e| e.into_inner());
        f(g.get_or_insert_with(Tracking::default))
    }

    /// A render target view at host handle `host` was made for PC resource `res`.
    pub fn rtv_made(host: usize, res: usize) {
        if mode() == MODE_TARGETS {
            with(|t| t.rtv.insert(host, res));
        }
    }

    /// Command list proxy `list` bound the render targets at host handles `hosts`.
    pub fn bound(list: usize, hosts: &[usize]) {
        if mode() == MODE_TARGETS {
            with(|t| {
                let rt = hosts.first().and_then(|h| t.rtv.get(h)).copied().unwrap_or(0);
                t.list_rt.insert(list, rt);
            });
        }
    }

    /// Command list proxy `list` recorded a draw.
    pub fn drew(list: usize) {
        if !targets_now() {
            return;
        }
        let frame = PRESENTED.load(Ordering::Relaxed) + 1;
        with(|t| {
            if t.frame != frame {
                (t.frame, t.draws, t.bytes, t.taken) = (frame, 0, 0, 0);
            }
            t.draws += 1;
            let rt = t.list_rt.get(&list).copied().unwrap_or(0);
            if rt != 0 {
                let draw = t.draws;
                t.dirty.insert(list, (rt, draw));
            }
        });
    }

    fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: a borrowed copy of the pointer, no reference taken; the barrier does
                    // not outlive `resource` and is never dropped as an owner.
                    pResource: unsafe { std::mem::transmute_copy(resource) },
                    Subresource: 0,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    /// A readback buffer for subresource 0 of a resource like `d`.
    fn readback(dev: &ID3D12Device, d: &D3D12_RESOURCE_DESC) -> windows::core::Result<(crate::mem_resource::TrackedResource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u64)> {
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
            Ok((crate::mem_resource::TrackedResource::readback(resource, total), fp, total))
        }
    }

    /// Records into `list` the copy of subresource 0 of `source` (in `state`) into `buf`.
    ///
    /// # Safety
    /// All objects are live; `source` is in `state` at this point of the list.
    unsafe fn record_copy(list: &ID3D12GraphicsCommandList, source: &ID3D12Resource, state: D3D12_RESOURCE_STATES, buf: &ID3D12Resource, fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
        // SAFETY: by the contract above; the copy locations borrow the pointers without a reference.
        unsafe {
            list.ResourceBarrier(&[transition(source, state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(buf), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(source), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE, state)]);
        }
    }

    /// Whether a resource like `d` can be copied out and decoded.
    fn copyable(d: &D3D12_RESOURCE_DESC) -> bool {
        d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.SampleDesc.Count <= 1 && decoder(d.Format.0).is_some()
    }

    fn warn_format(t: &mut Tracking, d: &D3D12_RESOURCE_DESC) {
        if !t.warned.contains(&d.Format.0) && t.warned.len() < 32 {
            t.warned.push(d.Format.0);
            xs_log!("capture: format {} (samples {}, dimension {}) is not written", d.Format.0, d.SampleDesc.Count, d.Dimension.0);
        }
    }

    /// Before command list proxy `list` (PC list `pc`) changes state: the target drawn into since
    /// the last copy, still a render target, is copied out.
    ///
    /// # Safety
    /// `pc` is the live PC command list of the proxy `list`, open for recording.
    pub unsafe fn flush(list: usize, pc: *mut c_void) {
        if mode() != MODE_TARGETS {
            return;
        }
        let Some((rt, draw)) = with(|t| t.dirty.remove(&list)) else { return };
        let frame = PRESENTED.load(Ordering::Relaxed) + 1;
        let rt_ptr = rt as *mut c_void;
        // SAFETY: `rt` is the PC resource of a render target view this list has bound and drawn
        // into, alive while the list records; `pc` by the contract above.
        let (Some(source), Some(cl)) = (unsafe { ID3D12Resource::from_raw_borrowed(&rt_ptr) }, unsafe { ID3D12GraphicsCommandList::from_raw_borrowed(&pc) }) else {
            return;
        };
        // SAFETY: plain queries on live objects.
        let d = unsafe { source.GetDesc() };
        if !copyable(&d) {
            with(|t| warn_format(t, &d));
            return;
        }
        let mut dev: Option<ID3D12Device> = None;
        // SAFETY: a plain query on a live resource.
        if unsafe { source.GetDevice(&mut dev) }.is_err() {
            return;
        }
        let Some(dev) = dev else { return };
        let Ok((buf, fp, size)) = readback(&dev, &d) else { return };
        let room = with(|t| {
            if t.bytes + size > FRAME_BUDGET || t.taken >= FRAME_TARGETS {
                return false;
            }
            t.bytes += size;
            t.taken += 1;
            true
        });
        if !room {
            return;
        }
        // SAFETY: the target is bound as render target 0 and only drawn into since: it is in RENDER_TARGET.
        unsafe { record_copy(cl, source, D3D12_RESOURCE_STATE_RENDER_TARGET, &buf, fp) };
        with(|t| t.pending.push(Pending { frame, draw, buf, fp, desc: d }));
    }

    /// The capture's own list and fence on the title's queue.
    struct Own {
        queue: usize,
        alloc: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        next: u64,
    }

    static OWN: Mutex<Option<Own>> = Mutex::new(None);

    fn own(queue: &ID3D12CommandQueue) -> windows::core::Result<std::sync::MutexGuard<'static, Option<Own>>> {
        let mut g = OWN.lock().unwrap_or_else(|e| e.into_inner());
        if g.as_ref().is_none_or(|o| o.queue != queue.as_raw() as usize) {
            // SAFETY: COM calls on a live queue and its device.
            unsafe {
                let mut dev: Option<ID3D12Device> = None;
                queue.GetDevice(&mut dev)?;
                let dev = dev.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))?;
                let kind = queue.GetDesc().Type;
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(kind)?;
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, kind, &alloc, None)?;
                list.Close()?;
                let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                *g = Some(Own { queue: queue.as_raw() as usize, alloc, list, fence, next: 0 });
            }
        }
        Ok(g)
    }

    /// Waits until the queue has done everything handed to it so far.
    fn drain(o: &mut Own, queue: &ID3D12CommandQueue) -> windows::core::Result<()> {
        o.next += 1;
        // SAFETY: COM calls on live objects; a null event makes the call wait.
        unsafe {
            queue.Signal(&o.fence, o.next)?;
            o.fence.SetEventOnCompletion(o.next, windows::Win32::Foundation::HANDLE::default())
        }
    }

    fn write(dir: &std::path::Path, name: &str, buf: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, d: &D3D12_RESOURCE_DESC) -> Result<(), String> {
        let mut at: *mut c_void = std::ptr::null_mut();
        // SAFETY: a readback buffer the GPU finished writing (the queue was drained).
        unsafe { buf.Map(0, None, Some(&mut at)) }.map_err(|e| e.to_string())?;
        let (w, h, pitch) = (d.Width as usize, d.Height as usize, fp.Footprint.RowPitch as usize);
        // SAFETY: the mapped buffer holds the whole footprint.
        let bytes = unsafe { std::slice::from_raw_parts(at.cast::<u8>(), pitch * h) };
        let rgba = decode(d.Format.0, bytes, w, h, pitch);
        // SAFETY: mapped above.
        unsafe { buf.Unmap(0, None) };
        let rgba = rgba.ok_or_else(|| format!("format {} not decoded", d.Format.0))?;
        write_png(&dir.join(name), &rgba, w as u32, h as u32).map_err(|e| e.to_string())
    }

    /// `PresentX` of PC resource `resource` on PC queue `queue`: the frame is counted, taken when
    /// picked, and the render targets of earlier picked frames are written.
    ///
    /// # Safety
    /// `queue` is a live PC `ID3D12CommandQueue`, `resource` null or a live PC `ID3D12Resource`
    /// in `RENDER_TARGET` (where a console title leaves the buffer it presents).
    pub unsafe fn presented(queue: *mut c_void, resource: *mut c_void) {
        if mode() == MODE_OFF {
            return;
        }
        let frame = PRESENTED.fetch_add(1, Ordering::Relaxed) + 1;
        let Some(cfg) = config() else { return };
        let pending: Vec<Pending> = with(|t| std::mem::take(&mut t.pending));
        let wanted = cfg.frames.wants(frame);
        if !wanted && pending.is_empty() {
            return;
        }
        // SAFETY: by the contract above.
        let Some(queue) = (unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }) else { return };
        let result = (|| -> Result<(), String> {
            let mut g = own(queue).map_err(|e| e.to_string())?;
            let Some(o) = g.as_mut() else { return Ok(()) };
            let mut shot = None;
            // SAFETY: by the contract above.
            if wanted && let Some(source) = unsafe { ID3D12Resource::from_raw_borrowed(&resource) } {
                // SAFETY: a plain query on a live resource.
                let d = unsafe { source.GetDesc() };
                if copyable(&d) {
                    let mut dev: Option<ID3D12Device> = None;
                    // SAFETY: COM calls on live objects; the copy runs on the title's queue before
                    // its present, with the buffer in RENDER_TARGET as the title left it.
                    unsafe {
                        source.GetDevice(&mut dev).map_err(|e| e.to_string())?;
                        let dev = dev.ok_or("no device")?;
                        let (buf, fp, _) = readback(&dev, &d).map_err(|e| e.to_string())?;
                        o.alloc.Reset().map_err(|e| e.to_string())?;
                        o.list.Reset(&o.alloc, None).map_err(|e| e.to_string())?;
                        record_copy(&o.list, source, D3D12_RESOURCE_STATE_RENDER_TARGET, &buf, fp);
                        o.list.Close().map_err(|e| e.to_string())?;
                        queue.ExecuteCommandLists(&[Some(o.list.cast().map_err(|e| e.to_string())?)]);
                        shot = Some((buf, fp, d));
                    }
                } else {
                    with(|t| warn_format(t, &d));
                }
            }
            drain(o, queue).map_err(|e| e.to_string())?;
            if let Some((buf, fp, d)) = shot {
                let name = format!("frame{frame}-presented.png");
                match write(&cfg.dir, &name, &buf, &fp, &d) {
                    Ok(()) => xs_log!("capture: {name} ({}x{}, format {})", d.Width, d.Height, d.Format.0),
                    Err(e) => xs_log!("capture: {name} not written: {e}"),
                }
            }
            let mut written = 0;
            for p in &pending {
                let name = format!("frame{}-draw{}-{}x{}.png", p.frame, p.draw, p.desc.Width, p.desc.Height);
                match write(&cfg.dir, &name, &p.buf, &p.fp, &p.desc) {
                    Ok(()) => written += 1,
                    Err(e) => xs_log!("capture: {name} not written: {e}"),
                }
            }
            if !pending.is_empty() {
                xs_log!("capture: {written} render target pictures of frame {} written", pending[0].frame);
            }
            Ok(())
        })();
        if let Err(e) = result {
            xs_log!("capture: frame {frame}: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_selection_parses_lists_and_the_powers_of_two() {
        for blank in [None, Some(""), Some("  "), Some("Pow2")] {
            assert_eq!(Frames::parse(blank), Ok(Frames::Pow2));
        }
        assert_eq!(Frames::parse(Some(" 600,3 ,3,,1")), Ok(Frames::List(vec![1, 3, 600])));
        assert!(Frames::parse(Some("4,five")).is_err());
        assert!(Frames::parse(Some("-1")).is_err());
    }

    #[test]
    fn frame_selection_answers_per_frame() {
        let powers = Frames::Pow2;
        assert!([1, 2, 4, 1024].iter().all(|&n| powers.wants(n)));
        assert!([0, 3, 6, 1000].iter().all(|&n| !powers.wants(n)));
        let list = Frames::List(vec![5, 9]);
        assert!(list.wants(5) && list.wants(9));
        assert!(!list.wants(1) && !list.wants(6));
    }

    #[test]
    fn eight_bit_formats_are_copied_or_swizzled() {
        // Two rows of two R8G8B8A8 pixels, 12 byte pitch with 4 bytes of padding per row.
        let bytes = [10, 20, 30, 40, 50, 60, 70, 80, 0xEE, 0xEE, 0xEE, 0xEE, 1, 2, 3, 4, 5, 6, 7, 8, 0xEE, 0xEE, 0xEE, 0xEE];
        assert_eq!(decode(28, &bytes, 2, 2, 12).unwrap(), vec![10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(decode(87, &[10, 20, 30, 40], 1, 1, 4).unwrap(), vec![30, 20, 10, 40]);
        assert_eq!(decode(88, &[10, 20, 30, 40], 1, 1, 4).unwrap(), vec![30, 20, 10, 255]);
        assert_eq!(decode(61, &[77], 1, 1, 1).unwrap(), vec![77, 77, 77, 255]);
        assert_eq!(decode(49, &[1, 2], 1, 1, 2).unwrap(), vec![1, 2, 0, 255]);
    }

    #[test]
    fn packed_and_float_formats_decode_to_known_colours() {
        // R10G10B10A2 with green full and alpha 2 of 3.
        let packed = (0x3FFu32 << 10) | (2 << 30);
        assert_eq!(decode(24, &packed.to_le_bytes(), 1, 1, 4).unwrap(), vec![0, 255, 0, 170]);
        // R16G16B16A16_FLOAT with 0.0, 1.0, 0.5 (sRGB 188) and 0.5 alpha (not curved).
        let halves: Vec<u8> = [0u16, 0x3C00, 0x3800, 0x3800].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(decode(10, &halves, 1, 1, 8).unwrap(), vec![0, 255, 188, 127]);
        // R32_FLOAT of 1.0 is white.
        assert_eq!(decode(41, &1.0f32.to_le_bytes(), 1, 1, 4).unwrap(), vec![255, 255, 255, 255]);
        // R11G11B10_FLOAT with red 1.0 and the rest zero.
        assert_eq!(decode(26, &(15u32 << 6).to_le_bytes(), 1, 1, 4).unwrap(), vec![255, 0, 0, 255]);
    }

    #[test]
    fn short_input_and_unknown_formats_are_refused() {
        assert_eq!(decode(28, &[0; 7], 2, 1, 8), None);
        assert_eq!(decode(28, &[0; 8], 2, 1, 7), None);
        assert_eq!(decode(0, &[0; 16], 1, 1, 16), None);
        assert_eq!(decode(28, &[], 0, 0, 0), Some(Vec::new()));
    }

    #[test]
    fn float_helpers_follow_the_formats() {
        assert_eq!(half(0x3C00), 1.0);
        assert_eq!(half(0xC000), -2.0);
        assert_eq!(half(0x7C00), f32::INFINITY);
        assert_eq!(half(0x0001), 2f32.powi(-24));
        assert_eq!(small_float(15 << 6, 6), 1.0);
        assert_eq!(small_float((16 << 6) | 32, 6), 3.0);
        assert_eq!(small_float(31 << 5, 5), f32::INFINITY);
        assert_eq!((srgb(f32::NAN), srgb(-1.0), srgb(2.0)), (0, 0, 255));
        assert_eq!((unorm(0, 2), unorm(1, 2), unorm(3, 2)), (0, 85, 255));
    }

    #[test]
    fn a_written_png_reads_back_with_the_same_pixels() {
        let dir = tempfile_dir("png");
        let path = dir.join("t.png");
        let pixels = [9, 8, 7, 6, 5, 4, 3, 2];
        write_png(&path, &pixels, 2, 1).unwrap();
        let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap())).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (2, 1));
        assert_eq!(&buf[..8], &pixels);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The whole path on a real PC device: a cleared render target is drawn into (as far as the
    /// capture can tell), flushed by the end of its list, then presented twice. Frames 1 and 2 are
    /// powers of two, so both are taken. Skips without a D3D12 device.
    #[cfg(windows)]
    #[test]
    fn presented_frames_and_their_targets_become_png_files() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::Interface;

        let dir = tempfile_dir("shots");
        let _ = std::fs::remove_dir_all(&dir);
        // As DECANT_CAPTURE=<dir> with DECANT_CAPTURE_TARGETS=1 would set it.
        std::fs::create_dir_all(&dir).unwrap();
        assert!(FORCED.set(Config { dir: dir.clone(), frames: Frames::Pow2, targets: true }).is_ok());
        STATE.store(MODE_TARGETS, Ordering::Relaxed);
        // SAFETY: plain D3D12 calls with valid descriptions; every object outlives its use.
        unsafe {
            // The system d3d12.dll by its path: this crate exports a D3D12CreateDevice of its own.
            let system = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into()) + r"\System32\d3d12.dll";
            let wide: Vec<u16> = system.encode_utf16().chain([0]).collect();
            let module = windows_sys::Win32::System::LibraryLoader::LoadLibraryW(wide.as_ptr());
            let create = windows_sys::Win32::System::LibraryLoader::GetProcAddress(module, c"D3D12CreateDevice".as_ptr().cast()).unwrap();
            let create: unsafe extern "system" fn(*mut std::ffi::c_void, i32, *const windows::core::GUID, *mut *mut std::ffi::c_void) -> i32 = std::mem::transmute(create);
            let mut raw = std::ptr::null_mut();
            if create(std::ptr::null_mut(), D3D_FEATURE_LEVEL_11_0.0, &ID3D12Device::IID, &mut raw) < 0 || raw.is_null() {
                eprintln!("skipped: no D3D12 device");
                return;
            }
            let dev = ID3D12Device::from_raw(raw);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 8,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut tex: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut tex).unwrap();
            let tex = tex.unwrap();
            let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let rtv = heap.GetCPUDescriptorHandleForHeapStart();
            dev.CreateRenderTargetView(&tex, None, rtv);
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.ClearRenderTargetView(rtv, &[1.0, 0.5, 0.0, 1.0], None);
            // What the proxies report for a title list: a view made, bound, a draw, the end of the list.
            let fake_list = 0x1234usize;
            rtv_made(rtv.ptr, tex.as_raw() as usize);
            bound(fake_list, &[rtv.ptr]);
            drew(fake_list);
            flush(fake_list, list.as_raw());
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            presented(queue.as_raw(), tex.as_raw());
            presented(queue.as_raw(), tex.as_raw());
        }
        let read = |name: &str| {
            let file = std::fs::File::open(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut reader = png::Decoder::new(std::io::BufReader::new(file)).read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buf).unwrap();
            ((info.width, info.height), buf[..4].to_vec())
        };
        // The clear colour; a GPU rounds 0.5 to 127 or 128.
        // The draw number counts every draw of the frame, those of tests running beside this one too.
        let target = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).find(|n| n.starts_with("frame1-draw") && n.ends_with("-8x4.png")).expect("the render target picture");
        for name in ["frame1-presented.png", "frame2-presented.png", target.as_str()] {
            let (size, px) = read(name);
            assert_eq!(size, (8, 4), "{name}");
            assert!(px[0] == 255 && (127..=128).contains(&px[1]) && px[2] == 0 && px[3] == 255, "{name}: {px:?}");
        }
    }

    /// A folder of its own per test: tests run at the same time.
    fn tempfile_dir(test: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("d3d12xs-capture-{}-{test}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
