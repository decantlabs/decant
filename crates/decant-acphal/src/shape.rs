//! The SHAPE context arrays (`docs/CONSOLE_LIBS.md` 2.4), the service thread that counts audio frames and the engine state the
//! `IAcpHal` commands change.
//!
//! The service thread counts audio frames and has the XMA engine (`xma`) fill the output rings of the enabled XMA contexts
//! every 2 ms, and at once when a command made room or enabled a context. The SRC, EQ, filter and DMA contexts are allocated
//! and zeroed, nothing converts or mixes them.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::abi::*;
use crate::heap::{Allocation, with_heaps};

/// Audio frames counted by the service thread since the process started (128 sample frames each).
static TICKS: AtomicU64 = AtomicU64::new(0);

/// The audio frame counter.
pub fn frame_counter() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

/// The audio clock in samples at 48 kHz (low 32 bits).
pub fn audio_clock() -> u32 {
    (frame_counter().wrapping_mul(u64::from(FRAME_SAMPLES))) as u32
}

struct Service {
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

/// The stop flag and condition variable of the running service thread, for `wake`.
type Signal = Arc<(Mutex<bool>, Condvar)>;

static RUNNING: Mutex<Option<Signal>> = Mutex::new(None);

/// Has the service thread run at once (a command made room in a ring or enabled a context).
pub fn wake() {
    if let Some(signal) = RUNNING.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        signal.1.notify_all();
    }
}

impl Service {
    fn start() -> Service {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let flag = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            decant_log::trace(&format!("AcpHal: the XMA service thread: {}", crate::priority::raise_current_thread()));
            let begin = Instant::now();
            let mut counted = 0u128;
            let (lock, cv) = &*flag;
            let mut stopped = lock.lock().unwrap_or_else(|e| e.into_inner());
            while !*stopped {
                // 48000 / 128 = 375 audio frames per second, counted from the elapsed time so a coarse timer loses none.
                let due = begin.elapsed().as_nanos() * 375 / 1_000_000_000;
                if due > counted {
                    TICKS.fetch_add((due - counted) as u64, Ordering::Relaxed);
                    counted = due;
                }
                drop(stopped);
                crate::xma::pump();
                stopped = lock.lock().unwrap_or_else(|e| e.into_inner());
                if *stopped {
                    break;
                }
                stopped = cv.wait_timeout(stopped, Duration::from_millis(2)).unwrap_or_else(|e| e.into_inner()).0;
            }
        });
        *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::clone(&stop));
        Service { stop, thread: Some(thread) }
    }

    fn stop(&mut self) {
        RUNNING.lock().unwrap_or_else(|e| e.into_inner()).take();
        {
            let mut stopped = self.stop.0.lock().unwrap_or_else(|e| e.into_inner());
            *stopped = true;
        }
        self.stop.1.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct ShapeState {
    /// Host addresses of the five allocations (SRC, EQ, filter, DMA, XMA with PCM), 0 when a count is 0.
    allocations: [usize; 5],
    counts: [u32; 6],
    service: Service,
}

static SHAPE: Mutex<Option<ShapeState>> = Mutex::new(None);

fn shape() -> MutexGuard<'static, Option<ShapeState>> {
    SHAPE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the contexts are allocated (the SHAPE engine runs).
pub fn is_allocated() -> bool {
    shape().is_some()
}

/// The counts of the allocated contexts.
pub fn allocated_counts() -> Option<[u32; 6]> {
    shape().as_ref().map(|s| s.counts)
}

/// `AcpHalAllocateShapeContexts`, on a `ShapeContexts` the caller owns.
pub fn allocate(contexts: &mut ShapeContexts) -> Hresult {
    let counts = contexts.counts;
    if counts.iter().zip(SHAPE_MAX_COUNTS).any(|(c, max)| *c > max) {
        return E_INVALIDARG;
    }
    if (counts[0] != 0 || counts[5] != 0) && counts[0] != counts[4] + counts[5] {
        return E_INVALIDARG;
    }
    let mut guard = shape();
    if guard.is_some() {
        return SHAPE_E_ALREADY_ALLOCATED;
    }
    // Five allocations in heap 2: XMA and PCM contexts share one, XMA first.
    let requests: [(u32, u32); 5] = [
        (counts[0] * SHAPE_ELEMENT_SIZES[0], SHAPE_ALIGNMENTS[0]),
        (counts[1] * SHAPE_ELEMENT_SIZES[1], SHAPE_ALIGNMENTS[1]),
        (counts[2] * SHAPE_ELEMENT_SIZES[2], SHAPE_ALIGNMENTS[2]),
        (counts[3] * SHAPE_ELEMENT_SIZES[3], SHAPE_ALIGNMENTS[3]),
        (counts[4] * SHAPE_ELEMENT_SIZES[4] + counts[5] * SHAPE_ELEMENT_SIZES[5], SHAPE_ALIGNMENTS[4]),
    ];
    let result = with_heaps(|heaps| {
        let mut done: Vec<Option<Allocation>> = Vec::new();
        for (size, alignment) in requests {
            if size == 0 {
                done.push(None);
                continue;
            }
            match heaps.alloc(size, alignment, 2) {
                Ok(a) => done.push(Some(a)),
                Err(e) => {
                    for a in done.iter().flatten() {
                        let _ = heaps.free(a.host);
                    }
                    return Err(e);
                }
            }
        }
        Ok(done)
    });
    let done = match result {
        Ok(Ok(done)) => done,
        Ok(Err(e)) | Err(e) => return e,
    };
    let at = |i: usize| done[i];
    let pairs: [Option<Allocation>; 6] = [
        at(0),
        at(1),
        at(2),
        at(3),
        at(4),
        at(4).map(|a| Allocation { host: a.host.wrapping_add((counts[4] * SHAPE_ELEMENT_SIZES[4]) as usize), apu: a.apu + counts[4] * SHAPE_ELEMENT_SIZES[4] }).filter(|_| counts[5] != 0),
    ];
    for (i, p) in pairs.iter().enumerate() {
        let valid = p.is_some() && counts[i] != 0;
        contexts.pointers[i] = if valid { p.map_or(0, |a| a.host as u64) } else { 0 };
        contexts.apu_addresses[i] = if valid { p.map_or(0, |a| a.apu) } else { 0 };
    }
    let mut allocations = [0usize; 5];
    for (slot, a) in allocations.iter_mut().zip(done.iter()) {
        *slot = a.map_or(0, |a| a.host as usize);
    }
    crate::xma::attach(pairs[4].map_or(0, |a| a.host as usize), counts[4]);
    *guard = Some(ShapeState { allocations, counts, service: Service::start() });
    S_OK
}

/// `AcpHalReleaseShapeContexts`.
pub fn release() -> Hresult {
    let mut guard = shape();
    let Some(mut state) = guard.take() else { return SHAPE_E_NOT_ALLOCATED };
    state.service.stop();
    crate::xma::detach();
    let _ = with_heaps(|heaps| {
        for host in state.allocations {
            if host != 0 {
                let _ = heaps.free(host as *const u8);
            }
        }
    });
    S_OK
}

/// What the commands changed in the engine.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Engine {
    pub config: u32,
    /// 512 bit mask of the enabled XMA contexts.
    pub xma_enabled: [u32; 16],
    /// XMA index to (value, APU address of the staging copy) of the last update.
    pub xma_update: HashMap<u32, (u32, u32)>,
    pub xma_input: HashMap<u32, u32>,
    /// DMA index to the blocks the title read, XMA index to the 256 byte output blocks it read.
    pub dma_consumed: HashMap<u32, u64>,
    pub xma_output_consumed: HashMap<u32, u64>,
    /// Mix buffers of 128 frames that a flowgraph cleared.
    pub mix_buffers: HashMap<u32, [i32; 128]>,
    /// Flowgraph operations that ran, and the ones that are open points.
    pub ops_clear: u64,
    pub ops_unimplemented: u64,
}

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

/// Runs `f` on the engine state.
pub fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    let mut guard = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(Engine::default))
}

/// A copy of the engine state.
pub fn engine_snapshot() -> Engine {
    with_engine(|e| e.clone())
}

/// Executes a flowgraph of `count` 64 bit commands read from `commands`. Returns the number of operations that ran.
pub fn run_flowgraph(commands: &[u64]) {
    with_engine(|engine| {
        for word in commands {
            match word & 0xFF {
                1 => {
                    let buffer = (word >> 33) as u32;
                    engine.mix_buffers.insert(buffer, [0; 128]);
                    engine.ops_clear += 1;
                }
                // 2 runs an SRC context and 0x10 copies a mix buffer to a DMA context: the context layouts are open.
                _ => engine.ops_unimplemented += 1,
            }
        }
    });
}
