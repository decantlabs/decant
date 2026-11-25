//! The `IAcpHal` object (`docs/CONSOLE_LIBS.md` 2.5): a C++ style object whose first field points at a table of 12 functions.
//!
//! Commands run synchronously inside `SubmitCommand`: the command is written to the next slot of the command ring in APU heap 1,
//! executed on the engine state of `shape`, and its slot is released. Messages go to a bounded ring and are dropped when it is full.
#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use decant_log::trace;

use crate::abi::*;
use crate::heap::{Allocation, with_heaps};
use crate::shape::{self, with_engine};
use crate::xma;

/// The table of 12 slots. There is no `QueryInterface`.
#[repr(C)]
pub struct IAcpHalVtbl {
    pub connect: unsafe extern "system" fn(*mut IAcpHal, u32, u32) -> Hresult,
    pub disconnect: unsafe extern "system" fn(*mut IAcpHal) -> Hresult,
    pub submit_command: unsafe extern "system" fn(*mut IAcpHal, u32, u64, u32, *const c_void, u32) -> Hresult,
    pub pop_message: unsafe extern "system" fn(*mut IAcpHal, *mut u8) -> i32,
    pub get_num_messages: unsafe extern "system" fn(*mut IAcpHal) -> u32,
    pub release: unsafe extern "system" fn(*mut IAcpHal) -> u32,
    pub unsupported: [unsafe extern "system" fn(*mut IAcpHal) -> Hresult; 4],
    pub map_driver_region: unsafe extern "system" fn(*mut IAcpHal, *mut u64) -> Hresult,
    pub destructor: unsafe extern "system" fn(*mut IAcpHal, u32) -> *mut IAcpHal,
}

/// The object: the table pointer first, as a C++ client expects.
#[repr(C)]
pub struct IAcpHal {
    pub vtbl: &'static IAcpHalVtbl,
    state: Mutex<HalState>,
}

pub static VTABLE: IAcpHalVtbl = IAcpHalVtbl {
    connect,
    disconnect,
    submit_command,
    pop_message,
    get_num_messages,
    release,
    unsupported: [unsupported, unsupported, unsupported, unsupported],
    map_driver_region,
    destructor,
};

static NEXT_ID: AtomicU32 = AtomicU32::new(1);
static CONNECTED: AtomicU32 = AtomicU32::new(0);

struct HalState {
    id: u32,
    connected: bool,
    command_slots: u32,
    command_ring: Option<Allocation>,
    message_ring: Option<Allocation>,
    message_capacity: usize,
    messages: VecDeque<AcpMessage>,
    /// Commands written to the ring and not yet executed, and the slot the next one uses.
    pending: u32,
    next_slot: u32,
    commands_run: u64,
    /// The audio frame clock of this connection: when it started and how many frame start messages it has queued.
    frames: FrameClock,
}

/// Audio frame start messages: one every 128 samples at 48 kHz (2.667 ms), counted from `Connect`. They are made when the
/// title looks at its messages, so a title that never reads them costs nothing. A title that comes back late gets at most
/// [`MAX_LATE_FRAMES`] of the frames it missed, the older ones are skipped.
struct FrameClock {
    start: Option<Instant>,
    queued: u64,
    /// The registered message mask from command 2. `None` means "every kind", which is what a command 2 without data asks.
    /// Before any command 2 no frame messages are sent.
    mask: Option<Option<u32>>,
}

/// Frame start messages a late reader gets at most.
pub const MAX_LATE_FRAMES: u64 = 8;

impl FrameClock {
    fn wanted(&self) -> bool {
        match self.mask {
            Some(None) => true,
            Some(Some(mask)) => mask & MSG_AUDIO_FRAME_START != 0,
            None => false,
        }
    }

    /// The frame starts due at `now` that were not queued yet, as frame numbers.
    fn due(&mut self, now: Instant) -> std::ops::Range<u64> {
        let Some(start) = self.start.filter(|_| self.wanted()) else { return 0..0 };
        let due = (now.saturating_duration_since(start).as_nanos() * u128::from(SAMPLE_RATE / FRAME_SAMPLES) / 1_000_000_000) as u64;
        let first = self.queued.max(due.saturating_sub(MAX_LATE_FRAMES));
        self.queued = due.max(self.queued);
        first..due
    }
}

// SAFETY: the raw pointers in `Allocation` point into the process wide heaps and are used under the lock of this state.
unsafe impl Send for HalState {}

impl HalState {
    fn push(&mut self, kind: u32, a: u32, b: u64) {
        if self.messages.len() >= self.message_capacity {
            return;
        }
        self.messages.push_back(AcpMessage { kind, clock: shape::audio_clock(), a, b, c: shape::frame_counter() as u32 });
    }

    /// Queues the frame start messages that are due: clock = frame × 128 samples, a = connection id, c = frame number.
    fn tick(&mut self) {
        if !self.connected {
            return;
        }
        let id = self.id;
        for frame in self.frames.due(Instant::now()) {
            if self.messages.len() >= self.message_capacity {
                break;
            }
            let clock = frame.wrapping_mul(u64::from(FRAME_SAMPLES)) as u32;
            self.messages.push_back(AcpMessage { kind: MSG_AUDIO_FRAME_START, clock, a: id, b: 0, c: frame as u32 });
        }
    }

    fn free_rings(&mut self) {
        let rings = [self.command_ring.take(), self.message_ring.take()];
        let _ = with_heaps(|heaps| {
            for ring in rings.into_iter().flatten() {
                let _ = heaps.free(ring.host);
            }
        });
    }

    fn disconnect(&mut self) -> bool {
        if !self.connected {
            return false;
        }
        self.connected = false;
        CONNECTED.fetch_sub(1, Ordering::SeqCst);
        let id = self.id;
        self.push(MSG_DISCONNECTED, id, 0);
        true
    }
}

impl Drop for IAcpHal {
    fn drop(&mut self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.disconnect();
        s.free_rings();
    }
}

/// A new object with a process unique id (`AcpHalCreate`).
pub fn create() -> *mut IAcpHal {
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let state = HalState {
        id,
        connected: false,
        command_slots: 0,
        command_ring: None,
        message_ring: None,
        message_capacity: DEFAULT_MESSAGE_SLOTS as usize,
        messages: VecDeque::new(),
        pending: 0,
        next_slot: 0,
        commands_run: 0,
        frames: FrameClock { start: None, queued: 0, mask: None },
    };
    Box::into_raw(Box::new(IAcpHal { vtbl: &VTABLE, state: Mutex::new(state) }))
}

impl IAcpHal {
    fn lock(&self) -> MutexGuard<'_, HalState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The process unique id of the object.
    pub fn id(&self) -> u32 {
        self.lock().id
    }

    pub fn is_connected(&self) -> bool {
        self.lock().connected
    }

    /// The APU addresses of the command and message rings, when allocated.
    pub fn ring_addresses(&self) -> (Option<u32>, Option<u32>) {
        let s = self.lock();
        (s.command_ring.map(|a| a.apu), s.message_ring.map(|a| a.apu))
    }

    /// Commands executed since creation.
    pub fn commands_run(&self) -> u64 {
        self.lock().commands_run
    }
}

unsafe extern "system" fn connect(this: *mut IAcpHal, command_slots: u32, message_slots: u32) -> Hresult {
    // SAFETY: the caller passes a live object.
    let mut s = unsafe { &*this }.lock();
    if s.connected {
        return HAL_E_ALREADY_CONNECTED;
    }
    if command_slots == 0 {
        return E_INVALIDARG;
    }
    if CONNECTED.load(Ordering::SeqCst) >= MAX_CONNECTED {
        return E_UNEXPECTED;
    }
    s.free_rings();
    s.messages.clear();
    let capacity = if message_slots == 0 { DEFAULT_MESSAGE_SLOTS } else { message_slots };
    let (Some(cmd_size), Some(msg_size)) = (command_slots.checked_mul(COMMAND_SLOT_SIZE), capacity.checked_mul(MESSAGE_SLOT_SIZE)) else {
        return E_OUTOFMEMORY;
    };
    let rings = with_heaps(|heaps| {
        let cmd = heaps.alloc(cmd_size, 0x100, 1)?;
        match heaps.alloc(msg_size, 0x100, 1) {
            Ok(msg) => Ok((cmd, msg)),
            Err(e) => {
                let _ = heaps.free(cmd.host);
                Err(e)
            }
        }
    });
    let (cmd, msg) = match rings {
        Ok(Ok(r)) => r,
        Ok(Err(e)) | Err(e) => return e,
    };
    s.command_ring = Some(cmd);
    s.message_ring = Some(msg);
    s.command_slots = command_slots;
    s.message_capacity = capacity as usize;
    s.pending = 0;
    s.next_slot = 0;
    s.frames = FrameClock { start: Some(Instant::now()), queued: 0, mask: None };
    s.connected = true;
    CONNECTED.fetch_add(1, Ordering::SeqCst);
    S_OK
}

unsafe extern "system" fn disconnect(this: *mut IAcpHal) -> Hresult {
    // SAFETY: the caller passes a live object.
    let mut s = unsafe { &*this }.lock();
    if s.disconnect() { S_OK } else { HAL_E_NOT_CONNECTED }
}

/// Number of payload words each implemented command type reads.
fn payload_words(kind: u32) -> Option<usize> {
    match kind {
        CMD_RUN_FLOWGRAPH => Some(2),
        CMD_CONFIGURE | CMD_XMA_ENABLE | CMD_XMA_DISABLE | CMD_XMA_SET_INPUT => Some(1),
        CMD_XMA_ENABLE_MASK | CMD_XMA_DISABLE_MASK => Some(16),
        CMD_XMA_UPDATE => Some(3),
        CMD_DMA_CONSUMED | CMD_XMA_OUTPUT_CONSUMED => Some(2),
        _ => None,
    }
}

/// Trace budget: the first lines of each call are logged with their result, then every 1000th.
fn traced(counter: &AtomicU32, first: u32) -> Option<u32> {
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
    (n <= first || n.is_multiple_of(1000)).then_some(n)
}

/// `SubmitCommand` with a trace line per command (type, id, frame, fence, result): the audio chain trace of issue #397 C.
unsafe extern "system" fn submit_command(this: *mut IAcpHal, kind: u32, id: u64, frame: u32, payload: *const c_void, fence: u32) -> Hresult {
    static CALLS: AtomicU32 = AtomicU32::new(0);
    // SAFETY: the arguments are the caller's, passed on unchanged.
    let hr = unsafe { submit_command_inner(this, kind, id, frame, payload, fence) };
    if let Some(n) = traced(&CALLS, 200) {
        trace(&format!(
            "AcpHal: SubmitCommand #{n} type {kind:#x} id {id:#x} frame {frame:#x} data {} fence {fence:#x} -> {:#010x}",
            if payload.is_null() { "none" } else { "yes" },
            hr as u32
        ));
    }
    hr
}

unsafe extern "system" fn submit_command_inner(this: *mut IAcpHal, kind: u32, id: u64, frame: u32, payload: *const c_void, fence: u32) -> Hresult {
    // SAFETY: the caller passes a live object.
    let mut s = unsafe { &*this }.lock();
    if !s.connected {
        return HAL_S_PENDING;
    }
    if kind == 0 || kind > CMD_TYPE_MAX {
        return E_INVALIDARG;
    }
    log_first_arrival(kind, id, frame, payload);
    // Type 4 and type 2 may come without data. The title sends type 2 with no data right after `Connect` and waits for its
    // acknowledgement.
    if payload.is_null() && kind != 4 && kind != CMD_CONFIGURE {
        return E_INVALIDARG;
    }
    let Some(count) = payload_words(kind) else {
        decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("AcpHal SubmitCommand type {kind:#x}"));
        trace(&format!("AcpHal: SubmitCommand type {kind:#x} is not implemented (E_NOTIMPL)"));
        return E_NOTIMPL;
    };
    if s.pending >= s.command_slots {
        return HAL_E_QUEUE_FULL;
    }
    let mut words = [0u32; 16];
    let count = if payload.is_null() { 0 } else { count };
    for (i, w) in words.iter_mut().enumerate().take(count) {
        // SAFETY: the caller's payload holds the words of this command type.
        *w = unsafe { std::ptr::read_unaligned((payload as *const u32).add(i)) };
    }
    // Write the command to its ring slot.
    if let Some(ring) = s.command_ring {
        let slot = s.next_slot;
        let mut image = [0u8; 24 + 64];
        image[0..4].copy_from_slice(&kind.to_le_bytes());
        image[8..16].copy_from_slice(&id.to_le_bytes());
        image[16..20].copy_from_slice(&frame.to_le_bytes());
        image[20..24].copy_from_slice(&fence.to_le_bytes());
        for (i, w) in words.iter().enumerate() {
            image[24 + i * 4..28 + i * 4].copy_from_slice(&w.to_le_bytes());
        }
        // SAFETY: the ring holds `command_slots` slots of COMMAND_SLOT_SIZE bytes and `slot` is below that.
        unsafe { std::ptr::copy_nonoverlapping(image.as_ptr(), ring.host.add((slot * COMMAND_SLOT_SIZE) as usize), image.len()) };
        s.next_slot = (slot + 1) % s.command_slots;
    }
    s.pending += 1;
    let result = execute(&mut s, kind, id, (!payload.is_null()).then_some(&words));
    s.pending -= 1;
    if result != S_OK {
        return result;
    }
    s.commands_run += 1;
    if fence != 0 {
        let _ = with_heaps(|heaps| {
            if let Some(p) = heaps.map_apu_range(fence, 4) {
                // SAFETY: four bytes inside a live allocation.
                unsafe { std::ptr::write_unaligned(p as *mut u32, 1) };
            }
        });
    }
    S_OK
}

/// Logs the first command of every type: its id, frame and whether it carries data (with the first four words).
fn log_first_arrival(kind: u32, id: u64, frame: u32, payload: *const c_void) {
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let bit = 1u32 << kind;
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    let data = if payload.is_null() {
        "no data".to_string()
    } else {
        let shown = payload_words(kind).unwrap_or(1).min(4);
        // SAFETY: a command with data carries at least one word, and a known type carries the words `payload_words` names.
        let words: Vec<String> = (0..shown).map(|i| format!("{:#x}", unsafe { std::ptr::read_unaligned((payload as *const u32).add(i)) })).collect();
        format!("data {}", words.join(" "))
    };
    trace(&format!("AcpHal: first command of type {kind:#x}: id {id:#x}, frame {frame:#x}, {data}"));
}

fn execute(s: &mut HalState, kind: u32, id: u64, words: Option<&[u32; 16]>) -> Hresult {
    let connection = s.id;
    let w = words.unwrap_or(&[0; 16]);
    match kind {
        CMD_RUN_FLOWGRAPH => {
            let (count, address) = (w[0], w[1]);
            let commands = (count != 0)
                .then(|| {
                    with_heaps(|heaps| {
                        let p = heaps.map_apu_range(address, count.checked_mul(8)?)?;
                        // SAFETY: `count * 8` bytes inside one live allocation.
                        Some((0..count as usize).map(|i| unsafe { std::ptr::read_unaligned((p as *const u64).add(i)) }).collect::<Vec<u64>>())
                    })
                    .ok()
                    .flatten()
                })
                .flatten();
            match commands {
                Some(list) => {
                    shape::run_flowgraph(&list);
                    s.push(MSG_FLOWGRAPH_COMPLETED, address, id);
                }
                None => s.push(MSG_FLOWGRAPH_TERMINATED, address, id),
            }
            return S_OK;
        }
        // With data, word 0 is the configuration value and the mask of the message kinds the title registers. Without data the
        // title registers every kind.
        CMD_CONFIGURE => match words {
            Some(w) => {
                with_engine(|e| e.config = w[0]);
                s.frames.mask = Some(Some(w[0]));
            }
            None => s.frames.mask = Some(None),
        },
        CMD_XMA_ENABLE | CMD_XMA_DISABLE => {
            if w[0] >= XMA_CONTEXT_COUNT_MAX {
                return E_INVALIDARG;
            }
            with_engine(|e| {
                let (word, bit) = ((w[0] / 32) as usize, 1u32 << (w[0] % 32));
                if kind == CMD_XMA_ENABLE {
                    e.xma_enabled[word] |= bit;
                } else {
                    e.xma_enabled[word] &= !bit;
                }
            });
            if kind == CMD_XMA_ENABLE {
                xma::enable(w[0]);
                shape::wake();
            } else {
                xma::disable(w[0]);
            }
        }
        CMD_XMA_ENABLE_MASK | CMD_XMA_DISABLE_MASK => {
            with_engine(|e| {
                for (dst, src) in e.xma_enabled.iter_mut().zip(w) {
                    if kind == CMD_XMA_ENABLE_MASK {
                        *dst |= src;
                    } else {
                        *dst &= !src;
                    }
                }
            });
            for index in (0..XMA_CONTEXT_COUNT_MAX).filter(|i| w[(i / 32) as usize] >> (i % 32) & 1 == 1) {
                if kind == CMD_XMA_ENABLE_MASK {
                    xma::enable(index);
                } else {
                    xma::disable(index);
                }
            }
            shape::wake();
        }
        CMD_XMA_UPDATE => {
            if w[0] >= XMA_CONTEXT_COUNT_MAX {
                return E_INVALIDARG;
            }
            with_engine(|e| e.xma_update.insert(w[0], (w[1], w[2])));
        }
        CMD_DMA_CONSUMED => {
            if w[0] >= DMA_CONTEXT_COUNT_MAX {
                return E_INVALIDARG;
            }
            with_engine(|e| *e.dma_consumed.entry(w[0]).or_default() += u64::from(w[1]));
        }
        CMD_XMA_OUTPUT_CONSUMED => {
            if w[0] >= XMA_CONTEXT_COUNT_MAX {
                return E_INVALIDARG;
            }
            with_engine(|e| *e.xma_output_consumed.entry(w[0]).or_default() += u64::from(w[1]));
            xma::output_consumed(w[0], w[1]);
            shape::wake();
        }
        CMD_XMA_SET_INPUT => {
            if w[0] >= XMA_CONTEXT_COUNT_MAX {
                return E_INVALIDARG;
            }
            // The words after the index are not specified: the index is all that is kept.
            with_engine(|e| e.xma_input.insert(w[0], 0));
        }
        _ => return E_NOTIMPL,
    }
    if id != 0 {
        s.push(MSG_COMMAND_COMPLETED, connection, id);
    }
    S_OK
}

/// `PopMessage` with a trace line per call: whether the title reads what we queued.
unsafe extern "system" fn pop_message(this: *mut IAcpHal, out: *mut u8) -> i32 {
    static CALLS: AtomicU32 = AtomicU32::new(0);
    // SAFETY: the arguments are the caller's, passed on unchanged.
    let got = unsafe { pop_message_inner(this, out) };
    if let Some(n) = traced(&CALLS, 40) {
        // SAFETY: when a message was copied, `out` holds its 28 bytes, the first word is the kind.
        let kind = if got != 0 { format!("{:#x}", unsafe { std::ptr::read_unaligned(out.cast::<u32>()) }) } else { "-".to_string() };
        trace(&format!("AcpHal: PopMessage #{n} -> {got}, kind {kind}"));
    }
    got
}

unsafe extern "system" fn pop_message_inner(this: *mut IAcpHal, out: *mut u8) -> i32 {
    // SAFETY: the caller passes a live object.
    let mut s = unsafe { &*this }.lock();
    if out.is_null() {
        return 0;
    }
    s.tick();
    match s.messages.pop_front() {
        Some(m) => {
            let bytes = m.to_bytes();
            // SAFETY: the caller's buffer holds an ACP_MESSAGE of 28 bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, MESSAGE_SIZE) };
            1
        }
        None => 0,
    }
}

/// `GetNumMessages` with a trace line per call.
unsafe extern "system" fn get_num_messages(this: *mut IAcpHal) -> u32 {
    static CALLS: AtomicU32 = AtomicU32::new(0);
    // SAFETY: the argument is the caller's, passed on unchanged.
    let n_msgs = unsafe { get_num_messages_inner(this) };
    if let Some(n) = traced(&CALLS, 40) {
        trace(&format!("AcpHal: GetNumMessages #{n} -> {n_msgs}"));
    }
    n_msgs
}

unsafe extern "system" fn get_num_messages_inner(this: *mut IAcpHal) -> u32 {
    // SAFETY: the caller passes a live object.
    let mut s = unsafe { &*this }.lock();
    s.tick();
    s.messages.len() as u32
}

unsafe extern "system" fn release(this: *mut IAcpHal) -> u32 {
    // SAFETY: the object came from `create` and the caller gives up its reference.
    drop(unsafe { Box::from_raw(this) });
    0
}

unsafe extern "system" fn unsupported(_this: *mut IAcpHal) -> Hresult {
    trace("AcpHal: an unidentified IAcpHal method was called (HAL_E_UNSUPPORTED)");
    HAL_E_UNSUPPORTED
}

unsafe extern "system" fn map_driver_region(_this: *mut IAcpHal, out: *mut u64) -> Hresult {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a writable UINT64 supplied by the caller.
    unsafe { *out = 0 };
    E_NOT_SUPPORTED
}

unsafe extern "system" fn destructor(this: *mut IAcpHal, flags: u32) -> *mut IAcpHal {
    if flags & 1 != 0 {
        // SAFETY: the object came from `create`.
        drop(unsafe { Box::from_raw(this) });
        return std::ptr::null_mut();
    }
    // SAFETY: the caller passes a live object.
    let mut s = unsafe { &*this }.lock();
    s.disconnect();
    s.free_rings();
    s.messages.clear();
    this
}
