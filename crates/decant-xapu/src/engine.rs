//! Connections, streams and the command queue of `xapu.dll` (`docs/CONSOLE_LIBS.md`, section 3).
//!
//! Commands run synchronously inside `enqueue`, in the order they arrive, and their results are committed at once, so results
//! come out in sequence order and the result event is signalled per command. The hardware runs streams in parallel, which a
//! caller cannot tell apart from this.
#![allow(unsafe_code)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use opus_decoder::OpusDecoder;
use decant_log::trace;

use crate::abi::*;
use crate::converter::{Converter, clamp_pitch};

/// What a started stream keeps between commands, plus reusable scratch buffers to avoid
/// per-command heap allocations on audio threads.
struct Config {
    channels: usize,
    start_pitch: f32,
    converter: Converter,
    decoder: Option<OpusDecoder>,
    scratch_pcm: Vec<f32>,
    scratch_planar: [Vec<f32>; 2],
    scratch_out: [Vec<f32>; 2],
}

#[derive(Default)]
struct StreamMeta {
    active: bool,
    /// Commands whose result has not been dequeued.
    outstanding: u32,
}

struct Committed {
    result: XapuResult,
    stream: u32,
    deactivate: bool,
}

struct Region {
    memory: *mut u8,
    layout: Option<Layout>,
}

// SAFETY: the region is plain memory owned by the connection and only freed on Drop.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Drop for Region {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            // SAFETY: allocated in `connect` with this layout.
            unsafe { dealloc(self.memory, layout) };
        }
    }
}

mod event {
    #[cfg(windows)]
    pub fn create() -> usize {
        // SAFETY: an unnamed auto reset event with default security.
        let event = unsafe { windows_sys::Win32::System::Threading::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) as usize };
        decant_log::handles::created(event, "XApu event");
        event
    }
    #[cfg(windows)]
    pub fn set(handle: usize) {
        // SAFETY: a handle this module created.
        unsafe { windows_sys::Win32::System::Threading::SetEvent(handle as _) };
    }
    #[cfg(windows)]
    pub fn close(handle: usize) {
        // SAFETY: a handle this module created.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(handle as _) };
    }
    #[cfg(not(windows))]
    pub fn create() -> usize {
        0
    }
    #[cfg(not(windows))]
    pub fn set(_: usize) {}
    #[cfg(not(windows))]
    pub fn close(_: usize) {}
}

struct QueueState {
    streams: Vec<StreamMeta>,
    sequence: u32,
    results: VecDeque<Committed>,
    dequeued: u32,
}

pub struct Connection {
    params: ConnectParams,
    _region: Region,
    event: usize,
    queue: Mutex<QueueState>,
    streams: Vec<Mutex<Option<Config>>>,
    pending_results: AtomicU32,
    dequeued_count: AtomicU32,
}

impl Drop for Connection {
    fn drop(&mut self) {
        if self.event != 0 {
            event::close(self.event);
        }
    }
}

#[derive(Default)]
struct Registry {
    connections: HashMap<Handle, Arc<Connection>>,
    streams_used: u32,
    next: Handle,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn registry() -> MutexGuard<'static, Option<Registry>> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

fn lookup(handle: Handle) -> Option<Arc<Connection>> {
    registry().as_ref()?.connections.get(&handle).cloned()
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Checks the parameters of `XApuConnect` against the ranges of the specification.
pub fn params_valid(p: &ConnectParams) -> bool {
    (1..=3).contains(&p.process_type)
        && (1..=MAX_STREAMS).contains(&p.stream_count)
        && (1..=MAX_QUEUE_DEPTH).contains(&p.queue_depth)
        && (p.process_type == PROCESS_DECODE_OPUS || (p.frame_count.is_multiple_of(2) && (0x80..=0x400).contains(&p.frame_count)))
        && p.options & !OPTION_MASK == 0
}

/// A connection made. `result_event` is 0 without option 8.
pub struct Connected {
    pub handle: Handle,
    pub memory: *mut u8,
    pub result_event: usize,
}

pub fn connect(p: &ConnectParams) -> Result<Connected, Hresult> {
    if !params_valid(p) {
        return Err(E_INVALIDARG);
    }
    if p.region_size > MAX_REGION_SIZE {
        return Err(E_OUTOFMEMORY);
    }
    let mut guard = registry();
    let reg = guard.get_or_insert_with(Registry::default);
    if reg.streams_used + p.stream_count > MAX_STREAMS {
        return Err(XAPU_E_NO_STREAMS);
    }
    let region = if p.region_size == 0 {
        Region { memory: std::ptr::null_mut(), layout: None }
    } else {
        let layout = Layout::from_size_align(p.region_size as usize, REGION_ALIGN).map_err(|_| E_OUTOFMEMORY)?;
        // SAFETY: a nonzero size.
        let memory = unsafe { alloc_zeroed(layout) };
        if memory.is_null() {
            return Err(E_OUTOFMEMORY);
        }
        Region { memory, layout: Some(layout) }
    };
    let result_event = if p.options & OPTION_RESULT_EVENT != 0 { event::create() } else { 0 };
    let memory = region.memory;
    let connection = Connection {
        params: *p,
        _region: region,
        event: result_event,
        queue: Mutex::new(QueueState {
            streams: (0..p.stream_count).map(|_| StreamMeta::default()).collect(),
            sequence: 0,
            results: VecDeque::new(),
            dequeued: 0,
        }),
        streams: (0..p.stream_count).map(|_| Mutex::new(None)).collect(),
        pending_results: AtomicU32::new(0),
        dequeued_count: AtomicU32::new(0),
    };
    reg.next += 1;
    let handle = 0x1000 + reg.next * 0x10;
    reg.streams_used += p.stream_count;
    reg.connections.insert(handle, Arc::new(connection));
    Ok(Connected { handle, memory, result_event })
}

pub fn disconnect(handle: Handle) -> Hresult {
    let mut guard = registry();
    let Some(reg) = guard.as_mut() else { return E_INVALIDARG };
    match reg.connections.remove(&handle) {
        Some(c) => {
            reg.streams_used -= c.params.stream_count;
            S_OK
        }
        None => E_INVALIDARG,
    }
}

/// Free streams and the total, as `XApuGetUtilization` reports them.
pub fn utilization() -> (u32, u32) {
    let used = registry().as_ref().map_or(0, |r| r.streams_used);
    (MAX_STREAMS - used, MAX_STREAMS)
}

pub fn is_connected(handle: Handle) -> bool {
    lookup(handle).is_some()
}

pub fn result_count(handle: Handle) -> (u32, u32) {
    lookup(handle).map_or((0, 0), |c| {
        (c.pending_results.load(Ordering::Acquire), c.dequeued_count.load(Ordering::Acquire))
    })
}

pub fn dequeue(handle: Handle) -> Result<XapuResult, Hresult> {
    let c = lookup(handle).ok_or(E_INVALIDARG)?;
    let committed = {
        let mut q = lock(&c.queue);
        let committed = q.results.pop_front().ok_or(XAPU_E_NO_RESULT)?;
        q.dequeued = q.dequeued.wrapping_add(1);
        c.pending_results.store(q.results.len() as u32, Ordering::Release);
        c.dequeued_count.store(q.dequeued, Ordering::Release);
        q.streams[committed.stream as usize].outstanding -= 1;
        committed
    };
    if committed.deactivate {
        // Clear the stream config before marking it inactive so a concurrent START cannot race.
        *lock(&c.streams[committed.stream as usize]) = None;
        lock(&c.queue).streams[committed.stream as usize].active = false;
    }
    Ok(committed.result)
}

/// The outcome of one command, before it is written to the result image.
#[derive(Default)]
struct Outcome {
    code: Hresult,
    state: u32,
    state_data: [u16; 4],
    frames_written: u32,
    bytes_consumed: u32,
    pitch: f32,
}

/// Enqueues `command` (already read from the caller) and returns the sequence number.
pub fn enqueue(handle: Handle, command: &Command) -> Result<u32, Hresult> {
    let kind = command.kind();
    if !matches!(kind, COMMAND_START | COMMAND_STOP | COMMAND_DATA | COMMAND_RESET) {
        return Err(E_INVALIDARG);
    }
    let c = lookup(handle).ok_or(E_INVALIDARG)?;
    let index = command.stream() as usize;
    let params = c.params;
    {
        let mut q = lock(&c.queue);
        let stream = q.streams.get_mut(index).ok_or(XAPU_E_BAD_STREAM)?;
        match (kind == COMMAND_START, stream.active) {
            (true, true) => return Err(XAPU_E_STREAM_ACTIVE),
            (false, false) => return Err(XAPU_E_BAD_STREAM),
            _ => {}
        }
        if stream.outstanding >= params.queue_depth {
            return Err(XAPU_E_QUEUE_FULL);
        }
        if kind == COMMAND_START && params.process_type != PROCESS_DECODE_OPUS {
            // The processing buffer holds the converter delay line: it must exceed the block of the first output by 1920 frames per channel.
            let channels = i64::from(command.channels());
            let spare = i64::from(command.processing_buffer_length() as i32) - channels * i64::from(params.frame_count) * 4;
            if spare < channels * 1920 {
                return Err(XAPU_E_BUFFER_TOO_SMALL);
            }
        }
        stream.outstanding += 1;
        if kind == COMMAND_START {
            stream.active = true;
        }
    }
    let began = Instant::now();
    let mut deactivate = false;
    let outcome = {
        let mut stream_cfg = lock(&c.streams[index]);
        match kind {
            COMMAND_START => start(&mut stream_cfg, &params, command),
            COMMAND_STOP => {
                deactivate = true;
                Outcome { pitch: stream_cfg.as_ref().map_or(0.0, |s| s.converter.pitch()), ..Default::default() }
            }
            COMMAND_RESET => reset(&mut stream_cfg, command),
            _ => data(&mut stream_cfg, &params, command),
        }
    };
    if outcome.code != S_OK {
        trace(&format!("xapu: command {kind} on stream {index} gave result {:#010x}", outcome.code));
    }
    let mut result = XapuResult::default();
    let b = &mut result.0;
    b[0..0x18].copy_from_slice(&command.0[0..0x18]);
    b[0x18..0x1C].copy_from_slice(&outcome.state.to_le_bytes());
    for (i, v) in outcome.state_data.iter().enumerate() {
        b[0x1C + 2 * i..0x1E + 2 * i].copy_from_slice(&v.to_le_bytes());
    }
    b[0x24..0x2C].copy_from_slice(&(began.elapsed().as_micros() as u64).to_le_bytes());
    b[0x2C..0x30].copy_from_slice(&outcome.code.to_le_bytes());
    b[0x30..0x34].copy_from_slice(&outcome.frames_written.to_le_bytes());
    b[0x34..0x38].copy_from_slice(&outcome.bytes_consumed.to_le_bytes());
    if kind == COMMAND_DATA {
        b[0x38..0x40].copy_from_slice(&command.0[0x38..0x40]);
    }
    b[0x40..0x44].copy_from_slice(&outcome.pitch.to_le_bytes());
    b[0x44..0x48].copy_from_slice(&command.0[0x28..0x2C]);
    b[0x48..0x50].copy_from_slice(&command.0[0x2C..0x34]);
    let sequence = {
        let mut q = lock(&c.queue);
        q.sequence = q.sequence.wrapping_add(1);
        q.results.push_back(Committed { result, stream: index as u32, deactivate });
        c.pending_results.store(q.results.len() as u32, Ordering::Release);
        q.sequence
    };
    if c.event != 0 {
        event::set(c.event);
    }
    Ok(sequence)
}

fn start(slot: &mut Option<Config>, params: &ConnectParams, command: &Command) -> Outcome {
    *slot = None;
    let channels = command.channels();
    let pitch = command.start_pitch();
    let convert = params.process_type != PROCESS_DECODE_OPUS;
    if !(1..=2).contains(&channels) || (convert && !pitch.is_finite()) {
        return Outcome { code: E_INVALIDARG, ..Default::default() };
    }
    let channels = channels as usize;
    let start_pitch = if pitch.is_finite() { clamp_pitch(pitch) } else { 1.0 };
    let decoder = if params.process_type == PROCESS_CONVERT_PCM {
        None
    } else {
        match OpusDecoder::new(48_000, channels) {
            Ok(d) => Some(d),
            Err(_) => return Outcome { code: E_FAIL, ..Default::default() },
        }
    };
    let scratch_pcm = decoder.as_ref().map_or_else(Vec::new, |d| vec![0.0f32; d.max_frame_size_per_channel() * channels]);
    *slot = Some(Config {
        channels,
        start_pitch,
        converter: Converter::new(channels, start_pitch),
        decoder,
        scratch_pcm,
        scratch_planar: [Vec::new(), Vec::new()],
        scratch_out: [Vec::new(), Vec::new()],
    });
    Outcome { pitch: start_pitch, ..Default::default() }
}

fn reset(slot: &mut Option<Config>, command: &Command) -> Outcome {
    let Some(config) = slot.as_mut() else { return Outcome { code: E_INVALIDARG, ..Default::default() } };
    config.converter.reset(config.start_pitch);
    if command.options() & COMMAND_OPTION_RESET_DECODER != 0
        && let Some(d) = config.decoder.as_mut()
    {
        d.reset();
    }
    Outcome { pitch: config.start_pitch, ..Default::default() }
}

fn fail(code: Hresult, pitch: f32) -> Outcome {
    Outcome { code, pitch, ..Default::default() }
}

fn data(slot: &mut Option<Config>, params: &ConnectParams, command: &Command) -> Outcome {
    let Some(config) = slot.as_mut() else { return fail(E_INVALIDARG, 0.0) };
    let pitch = config.converter.pitch();
    let out_ptr = command.output_pointer() as usize as *mut f32;
    let in_ptr = command.input_pointer() as usize as *const u8;
    let input_len = command.input_length() as usize;
    if out_ptr.is_null() {
        return fail(E_INVALIDARG, pitch);
    }
    let channels = config.channels;
    let capacity = command.output_capacity() as usize;
    let deinterleaved_out = params.options & OPTION_DEINTERLEAVED_OUTPUT != 0;
    if params.process_type == PROCESS_DECODE_OPUS {
        let Some(n) = decode_packet(config, in_ptr, input_len) else { return fail(XAPU_E_BAD_OPUS_PACKET, pitch) };
        if n > capacity {
            return fail(XAPU_E_BUFFER_TOO_SMALL, pitch);
        }
        let saturated = write_planar(out_ptr, &config.scratch_planar[..channels], 0, n, capacity, deinterleaved_out);
        return Outcome {
            state: STATE_OUTPUT_READY | if saturated { STATE_SATURATION } else { 0 },
            frames_written: n as u32,
            bytes_consumed: command.input_length(),
            pitch,
            ..Default::default()
        };
    }

    // Convert paths: fill `config.scratch_planar[..channels]` without reallocating.
    let flush = command.options() & COMMAND_OPTION_FLUSH != 0;
    let block = params.frame_count as usize;
    let input_frames: usize;
    let mut consumed = command.input_length();
    if params.process_type == PROCESS_DECODE_OPUS_CONVERT {
        match decode_packet(config, in_ptr, input_len) {
            Some(n) => input_frames = n,
            None => return fail(XAPU_E_BAD_OPUS_PACKET, pitch),
        }
    } else {
        let frames = command.frame_count() as usize;
        let interleaved_in = params.options & OPTION_DEINTERLEAVED_INPUT == 0;
        let step = if channels == 2 && interleaved_in { 8 } else { 4 };
        if !(4..=960).contains(&frames) || !frames.is_multiple_of(step) || in_ptr.is_null() || input_len < frames * channels * 4 {
            return fail(E_INVALIDARG, pitch);
        }
        let in_f32 = in_ptr as *const f32;
        for c in 0..channels {
            config.scratch_planar[c].clear();
            config.scratch_planar[c].reserve(frames);
        }
        if interleaved_in {
            if channels == 1 {
                for f in 0..frames {
                    // SAFETY: `in_ptr` holds `frames` floats, at any alignment.
                    config.scratch_planar[0].push(unsafe { std::ptr::read_unaligned(in_f32.add(f)) });
                }
            } else {
                let (c0, c1) = config.scratch_planar.split_at_mut(1);
                for f in 0..frames {
                    // SAFETY: `in_ptr` holds `frames * 2` floats, at any alignment.
                    unsafe {
                        c0[0].push(std::ptr::read_unaligned(in_f32.add(f * 2)));
                        c1[0].push(std::ptr::read_unaligned(in_f32.add(f * 2 + 1)));
                    }
                }
            }
        } else {
            for c in 0..channels {
                for f in 0..frames {
                    // SAFETY: `in_ptr` holds `frames * channels` floats, at any alignment.
                    config.scratch_planar[c].push(unsafe { std::ptr::read_unaligned(in_f32.add(c * frames + f)) });
                }
            }
        }
        input_frames = frames;
        consumed = (frames * channels * 4) as u32;
    }
    if capacity < block && !flush {
        return fail(XAPU_E_BUFFER_TOO_SMALL, pitch);
    }
    let (target, ramp) = (command.target_pitch(), command.pitch_ramp());
    if !target.is_finite() || !ramp.is_finite() {
        return fail(E_INVALIDARG, pitch);
    }
    {
        let slices: [&[f32]; 2] = [config.scratch_planar[0].as_slice(), config.scratch_planar[1].as_slice()];
        config.converter.push(&slices[..channels], target, ramp);
    }
    if flush {
        config.converter.flush(target, ramp);
    }
    let available = config.converter.pending_frames();
    let n = if flush { available.min(capacity) } else { (available / block * block).min(capacity / block * block) };
    for c in 0..channels {
        config.scratch_out[c].clear();
    }
    config.converter.take(n, &mut config.scratch_out[..channels]);
    let saturated = write_planar(out_ptr, &config.scratch_out[..channels], 0, n, capacity, deinterleaved_out);
    let left = config.converter.pending_frames();
    let more = if flush { left > 0 } else { left >= block };
    let unconverted = config.converter.unconverted_frames();
    if flush && !more {
        // The segment is over: the next data starts a new one, with the pitch it reached.
        config.converter.restart();
    }
    let state = if n > 0 { STATE_OUTPUT_READY } else { 0 } | if more { STATE_MORE_OUTPUT } else { STATE_NEED_MORE_INPUT } | if saturated { STATE_SATURATION } else { 0 };
    Outcome {
        code: S_OK,
        state,
        state_data: [left.min(0xFFFF) as u16, input_frames.min(0xFFFF) as u16, 0, unconverted.min(0xFFFF) as u16],
        frames_written: n as u32,
        bytes_consumed: consumed,
        pitch: config.converter.pitch(),
    }
}

/// Decodes the Opus packet at `ptr` into `config.scratch_planar`, returning the decoded frame count.
fn decode_packet(config: &mut Config, ptr: *const u8, len: usize) -> Option<usize> {
    if ptr.is_null() || len == 0 {
        return None;
    }
    let decoder = config.decoder.as_mut()?;
    // SAFETY: the caller's input holds `len` bytes.
    let packet = unsafe { std::slice::from_raw_parts(ptr, len) };
    let channels = config.channels;
    let needed = decoder.max_frame_size_per_channel() * channels;
    if config.scratch_pcm.len() < needed {
        config.scratch_pcm.resize(needed, 0.0);
    }
    let frames = decoder.decode_float(packet, &mut config.scratch_pcm, false).ok()?;
    for c in 0..channels {
        config.scratch_planar[c].clear();
        config.scratch_planar[c].reserve(frames);
    }
    if channels == 1 {
        config.scratch_planar[0].extend_from_slice(&config.scratch_pcm[..frames]);
    } else {
        let (c0, c1) = config.scratch_planar.split_at_mut(1);
        for pair in config.scratch_pcm[..frames * 2].as_chunks::<2>().0 {
            c0[0].push(pair[0]);
            c1[0].push(pair[1]);
        }
    }
    Some(frames)
}

/// Writes frames `[from, from + n)` of `planar` to `out`, interleaved or, when `planar_out`, one channel after the other at the
/// capacity. Non finite values become 0. Returns whether a sample exceeded full scale.
fn write_planar(out: *mut f32, planar: &[Vec<f32>], from: usize, n: usize, capacity: usize, planar_out: bool) -> bool {
    let channels = planar.len();
    let mut saturated = false;
    if !planar_out && channels == 2 {
        let c0 = &planar[0][from..from + n];
        let c1 = &planar[1][from..from + n];
        for (f, (&s0, &s1)) in c0.iter().zip(c1.iter()).enumerate() {
            let v0 = if s0.is_finite() { s0 } else { 0.0 };
            let v1 = if s1.is_finite() { s1 } else { 0.0 };
            saturated |= v0.abs() > 1.0 || v1.abs() > 1.0;
            // SAFETY: the caller's output holds `capacity` frames of `2` floats.
            unsafe {
                std::ptr::write_unaligned(out.add(f * 2), v0);
                std::ptr::write_unaligned(out.add(f * 2 + 1), v1);
            }
        }
        return saturated;
    }
    for (c, chan) in planar.iter().enumerate() {
        for (f, &raw) in chan[from..from + n].iter().enumerate() {
            let v = if raw.is_finite() { raw } else { 0.0 };
            saturated |= v.abs() > 1.0;
            let index = if planar_out { c * capacity + f } else { f * channels + c };
            // SAFETY: the caller's output holds `capacity` frames of `channels` floats.
            unsafe { std::ptr::write_unaligned(out.add(index), v) };
        }
    }
    saturated
}
