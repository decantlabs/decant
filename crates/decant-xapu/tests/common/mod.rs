//! A C style client of the `xapu` exports.
#![allow(unsafe_code, dead_code)]

use std::sync::{Mutex, MutexGuard};

use xapu::abi::*;
use xapu::exports::*;

static SERIAL: Mutex<()> = Mutex::new(());

/// The stream budget is process wide: tests run one after another.
pub fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn params(process_type: u32, streams: u32, depth: u32, region: u32, frames: u32, options: u32) -> ConnectParams {
    ConnectParams { process_type, stream_count: streams, queue_depth: depth, region_size: region, frame_count: frames, options }
}

/// Calls `XApuConnect`, returns the result, the handle and the region.
pub fn try_connect(p: &ConnectParams) -> (Hresult, Handle, ConnectionRegion) {
    let mut region = ConnectionRegion { memory: std::ptr::dangling_mut(), result_event: 0x77 };
    let mut handle: Handle = 0x99;
    // SAFETY: writable out structures.
    let r = unsafe { XApuConnect(p, &mut region, &mut handle) };
    (r, handle, region)
}

/// A connection that disconnects when it goes out of scope.
pub struct Conn {
    pub handle: Handle,
    pub region: ConnectionRegion,
}

impl Conn {
    pub fn open(p: ConnectParams) -> Conn {
        let (r, handle, region) = try_connect(&p);
        assert_eq!(r, S_OK, "{p:?}");
        assert_ne!(handle, 0);
        Conn { handle, region }
    }

    pub fn enqueue(&self, c: &Command) -> (Hresult, u32) {
        let mut seq = 0xAAAA_u32;
        // SAFETY: a command of 0x40 bytes and a writable sequence.
        let r = unsafe { XApuEnqueueCommand(self.handle, c, &mut seq) };
        (r, seq)
    }

    pub fn dequeue(&self) -> (Hresult, XapuResult) {
        let mut r = XapuResult([0xCD; RESULT_SIZE]);
        // SAFETY: a writable result of 0x50 bytes.
        let code = unsafe { XApuDequeueResult(self.handle, &mut r) };
        (code, r)
    }

    /// Dequeues one result that must exist.
    pub fn result(&self) -> XapuResult {
        let (code, r) = self.dequeue();
        assert_eq!(code, S_OK);
        r
    }

    pub fn counts(&self) -> (u32, u32) {
        let mut dequeued = 0xFFFF;
        // SAFETY: a writable counter.
        let n = unsafe { XApuGetResultCount(self.handle, &mut dequeued) };
        (n, dequeued)
    }

    /// Starts a stream and dequeues the result, which must be a success.
    pub fn start(&self, stream: u32, channels: u32, pitch: f32, buffer: u32) -> XapuResult {
        let (r, _) = self.enqueue(&Command::new(COMMAND_START, stream, 0).with_start(channels, pitch, buffer));
        assert_eq!(r, S_OK);
        let res = self.result();
        assert_eq!(res.code(), S_OK);
        res
    }

    /// A PCM or Opus data command, run to its result. `input` is bytes, `out` receives `capacity * channels` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn data(&self, stream: u32, options: u32, ramp: f32, target: f32, frames: u32, input: &[u8], capacity: u32, out: &mut [f32]) -> (Hresult, XapuResult) {
        let c = Command::new(COMMAND_DATA, stream, options).with_data(ramp, target, frames, input.len() as u32, input.as_ptr() as u64, capacity, out.as_mut_ptr() as u64);
        let (r, _) = self.enqueue(&c);
        if r != S_OK {
            return (r, XapuResult::default());
        }
        (r, self.result())
    }
}

impl Drop for Conn {
    fn drop(&mut self) {
        assert_eq!(XApuDisconnect(self.handle), S_OK);
    }
}

pub fn bytes_of(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// A CELT only, full band, 20 ms Opus packet of pseudo random content: any payload decodes to 960 frames.
pub fn opus_packet(stereo: bool, seed: u32) -> Vec<u8> {
    let mut p = vec![if stereo { 0xFC } else { 0xF8 }];
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    for _ in 0..60 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        p.push((x >> 24) as u8);
    }
    p
}
