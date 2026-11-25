//! Connections, streams and the command queue of `xapu.dll` (`docs/CONSOLE_LIBS.md` 3.4).
//!
//! Commands run synchronously inside `enqueue`, in the order they arrive, and their results are committed at once, so results
//! come out in sequence order and the result event is signalled per command. The hardware runs streams in parallel, which a
//! caller cannot tell apart from this.
#![allow(unsafe_code)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use opus_decoder::OpusDecoder;
use decant_log::trace;

use crate::abi::*;
use crate::converter::{Converter, clamp_pitch};

/// What a started stream keeps between commands.
struct Config {
    channels: usize,
    start_pitch: f32,
    converter: Converter,
    decoder: Option<OpusDecoder>,
}

#[derive(Default)]
struct Stream {
    active: bool,
    /// Commands whose result has not been dequeued.
    outstanding: u32,
    config: Option<Config>,
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

// SAFETY: the region is plain memory owned by the connection.
unsafe impl Send for Region {}

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

pub struct Connection {
    params: ConnectParams,
    _region: Region,
    event: usize,
    streams: Vec<Stream>,
    sequence: u32,
    results: VecDeque<Committed>,
    dequeued: u32,
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
    connections: HashMap<Handle, Arc<Mutex<Connection>>>,
    streams_used: u32,
    next: Handle,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn registry() -> MutexGuard<'static, Option<Registry>> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

fn lookup(handle: Handle) -> Option<Arc<Mutex<Connection>>> {
    registry().as_ref()?.connections.get(&handle).cloned()
}

fn lock(c: &Arc<Mutex<Connection>>) -> MutexGuard<'_, Connection> {
    c.lock().unwrap_or_else(|e| e.into_inner())
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
        streams: (0..p.stream_count).map(|_| Stream::default()).collect(),
        sequence: 0,
        results: VecDeque::new(),
        dequeued: 0,
    };
    reg.next += 1;
    let handle = 0x1000 + reg.next * 0x10;
    reg.streams_used += p.stream_count;
    reg.connections.insert(handle, Arc::new(Mutex::new(connection)));
    Ok(Connected { handle, memory, result_event })
}

pub fn disconnect(handle: Handle) -> Hresult {
    let mut guard = registry();
    let Some(reg) = guard.as_mut() else { return E_INVALIDARG };
    match reg.connections.remove(&handle) {
        Some(c) => {
            reg.streams_used -= lock(&c).params.stream_count;
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
        let c = lock(&c);
        (c.results.len() as u32, c.dequeued)
    })
}

pub fn dequeue(handle: Handle) -> Result<XapuResult, Hresult> {
    let c = lookup(handle).ok_or(E_INVALIDARG)?;
    let mut c = lock(&c);
    let committed = c.results.pop_front().ok_or(XAPU_E_NO_RESULT)?;
    c.dequeued = c.dequeued.wrapping_add(1);
    let stream = &mut c.streams[committed.stream as usize];
    stream.outstanding -= 1;
    if committed.deactivate {
        stream.active = false;
        stream.config = None;
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
    let c = lookup(handle).ok_or(E_INVALIDARG)?;
    let mut c = lock(&c);
    let kind = command.kind();
    if !matches!(kind, COMMAND_START | COMMAND_STOP | COMMAND_DATA | COMMAND_RESET) {
        return Err(E_INVALIDARG);
    }
    let index = command.stream() as usize;
    let params = c.params;
    let stream = c.streams.get(index).ok_or(XAPU_E_BAD_STREAM)?;
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
    c.sequence = c.sequence.wrapping_add(1);
    let sequence = c.sequence;
    let began = Instant::now();
    let mut deactivate = false;
    let outcome = {
        let c = &mut *c;
        let stream = &mut c.streams[index];
        match kind {
            COMMAND_START => {
                stream.active = true;
                start(stream, &c.params, command)
            }
            COMMAND_STOP => {
                deactivate = true;
                Outcome { pitch: stream.config.as_ref().map_or(0.0, |s| s.converter.pitch()), ..Default::default() }
            }
            COMMAND_RESET => reset(stream, command),
            _ => data(stream, &c.params, command),
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
    c.streams[index].outstanding += 1;
    c.results.push_back(Committed { result, stream: index as u32, deactivate });
    if c.event != 0 {
        event::set(c.event);
    }
    Ok(sequence)
}

fn start(stream: &mut Stream, params: &ConnectParams, command: &Command) -> Outcome {
    stream.config = None;
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
    stream.config = Some(Config { channels, start_pitch, converter: Converter::new(channels, start_pitch), decoder });
    Outcome { pitch: start_pitch, ..Default::default() }
}

fn reset(stream: &mut Stream, command: &Command) -> Outcome {
    let Some(config) = stream.config.as_mut() else { return Outcome { code: E_INVALIDARG, ..Default::default() } };
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

fn data(stream: &mut Stream, params: &ConnectParams, command: &Command) -> Outcome {
    let Some(config) = stream.config.as_mut() else { return fail(E_INVALIDARG, 0.0) };
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
        let Some(planar) = decode_packet(config, in_ptr, input_len) else { return fail(XAPU_E_BAD_OPUS_PACKET, pitch) };
        let n = planar[0].len();
        if n > capacity {
            return fail(XAPU_E_BUFFER_TOO_SMALL, pitch);
        }
        let saturated = write_planar(out_ptr, &planar, 0, n, capacity, deinterleaved_out);
        return Outcome {
            state: STATE_OUTPUT_READY | if saturated { STATE_SATURATION } else { 0 },
            frames_written: n as u32,
            bytes_consumed: command.input_length(),
            pitch,
            ..Default::default()
        };
    }

    // Convert paths: get the input as planar frames.
    let flush = command.options() & COMMAND_OPTION_FLUSH != 0;
    let block = params.frame_count as usize;
    let planar: Vec<Vec<f32>>;
    let mut consumed = command.input_length();
    if params.process_type == PROCESS_DECODE_OPUS_CONVERT {
        match decode_packet(config, in_ptr, input_len) {
            Some(p) => planar = p,
            None => return fail(XAPU_E_BAD_OPUS_PACKET, pitch),
        }
    } else {
        let frames = command.frame_count() as usize;
        let interleaved_in = params.options & OPTION_DEINTERLEAVED_INPUT == 0;
        let step = if channels == 2 && interleaved_in { 8 } else { 4 };
        if !(4..=960).contains(&frames) || !frames.is_multiple_of(step) || in_ptr.is_null() || input_len < frames * channels * 4 {
            return fail(E_INVALIDARG, pitch);
        }
        // SAFETY: the caller's input holds `frames * channels` floats, at any alignment.
        let samples: Vec<f32> = (0..frames * channels).map(|i| unsafe { std::ptr::read_unaligned((in_ptr as *const f32).add(i)) }).collect();
        planar = (0..channels)
            .map(|c| (0..frames).map(|f| if interleaved_in { samples[f * channels + c] } else { samples[c * frames + f] }).collect())
            .collect();
        consumed = (frames * channels * 4) as u32;
    }
    if capacity < block && !flush {
        return fail(XAPU_E_BUFFER_TOO_SMALL, pitch);
    }
    let (target, ramp) = (command.target_pitch(), command.pitch_ramp());
    if !target.is_finite() || !ramp.is_finite() {
        return fail(E_INVALIDARG, pitch);
    }
    let input_frames = planar[0].len();
    let slices: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
    config.converter.push(&slices, target, ramp);
    if flush {
        config.converter.flush(target, ramp);
    }
    let available = config.converter.pending_frames();
    let n = if flush { available.min(capacity) } else { (available / block * block).min(capacity / block * block) };
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); channels];
    config.converter.take(n, &mut out);
    let saturated = write_planar(out_ptr, &out, 0, n, capacity, deinterleaved_out);
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

/// Decodes the Opus packet at `ptr`, as planar floats. `None` when it is not a packet.
fn decode_packet(config: &mut Config, ptr: *const u8, len: usize) -> Option<Vec<Vec<f32>>> {
    if ptr.is_null() || len == 0 {
        return None;
    }
    let decoder = config.decoder.as_mut()?;
    // SAFETY: the caller's input holds `len` bytes.
    let packet = unsafe { std::slice::from_raw_parts(ptr, len) };
    let channels = config.channels;
    let mut pcm = vec![0.0f32; decoder.max_frame_size_per_channel() * channels];
    let frames = decoder.decode_float(packet, &mut pcm, false).ok()?;
    let planar = (0..channels).map(|c| (0..frames).map(|f| pcm[f * channels + c]).collect()).collect();
    Some(planar)
}

/// Writes frames `[from, from + n)` of `planar` to `out`, interleaved or, when `planar_out`, one channel after the other at the
/// capacity. Non finite values become 0. Returns whether a sample exceeded full scale.
fn write_planar(out: *mut f32, planar: &[Vec<f32>], from: usize, n: usize, capacity: usize, planar_out: bool) -> bool {
    let channels = planar.len();
    let mut saturated = false;
    for (c, chan) in planar.iter().enumerate() {
        for f in 0..n {
            let mut v = chan[from + f];
            if !v.is_finite() {
                v = 0.0;
            }
            saturated |= v.abs() > 1.0;
            let index = if planar_out { c * capacity + f } else { f * channels + c };
            // SAFETY: the caller's output holds `capacity` frames of `channels` floats.
            unsafe { std::ptr::write_unaligned(out.add(index), v) };
        }
    }
    saturated
}
