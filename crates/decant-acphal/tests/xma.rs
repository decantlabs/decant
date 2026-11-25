//! The XMA engine driven as a title drives it: a context in the shared array, input and output in APU memory, commands
//! through `IAcpHal`. The streams are synthetic (`decant_xma::synth`).
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use AcpHal::abi::*;
use AcpHal::exports::*;
use AcpHal::hal::IAcpHal;
use AcpHal::xma;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: set before any test of this process decodes, read once by the monitor.
    unsafe { std::env::set_var("DECANT_XMA_MUTE", "1") };
    let _ = ApuCreateHeap(0x400_0000, 0);
    guard
}

/// APU memory: (host pointer, APU address).
fn apu(bytes: u32, alignment: u32) -> (*mut u8, u32) {
    let (mut va, mut apu) = (std::ptr::null_mut(), 0);
    // SAFETY: writable out values.
    assert_eq!(unsafe { ApuAlloc(&mut va, &mut apu, bytes, alignment, 1) }, S_OK);
    (va as *mut u8, apu)
}

struct Setup {
    hal: *mut IAcpHal,
    words: *mut u32,
    input: (*mut u8, u32),
    output: (*mut u8, u32),
}

const INDEX: u32 = 2;
const RING: u32 = 4;

impl Setup {
    /// Contexts allocated, `stream` in input buffer 0, a ring of 4 blocks, 48 kHz mono, nothing enabled yet.
    fn new(stream: &[u8]) -> Setup {
        let mut c = ShapeContexts { counts: [0, 0, 0, 0, 4, 0], ..Default::default() };
        // SAFETY: a writable structure.
        assert_eq!(unsafe { AcpHalAllocateShapeContexts(&mut c) }, S_OK);
        let words = (c.pointers[4] as *mut u32).wrapping_add(INDEX as usize * 16);
        let input = apu(stream.len() as u32, 2048);
        // SAFETY: the allocation holds the stream.
        unsafe { std::ptr::copy_nonoverlapping(stream.as_ptr(), input.0, stream.len()) };
        let output = apu(RING * 256, 256);
        // SAFETY: the ring.
        unsafe { output.0.write_bytes(0x55, (RING * 256) as usize) };
        let mut hal = std::ptr::null_mut();
        // SAFETY: a writable out pointer, then a live object.
        unsafe {
            assert_eq!(AcpHalCreate(&mut hal), S_OK);
            assert_eq!(((*hal).vtbl.connect)(hal, 64, 64), S_OK);
        }
        let s = Setup { hal, words, input, output };
        let packets = (stream.len() / 2048) as u32;
        s.set(0, packets | 1 << 20 | RING << 22);
        s.set(1, 1 << 31 | 3 << 27);
        s.set(5, input.1);
        s.set(7, output.1);
        s
    }

    fn get(&self, w: usize) -> u32 {
        // SAFETY: inside the context.
        unsafe { self.words.add(w).read_volatile() }
    }

    fn set(&self, w: usize, v: u32) {
        // SAFETY: inside the context.
        unsafe { self.words.add(w).write_volatile(v) }
    }

    fn field(&self, w: usize, shift: u32, bits: u32) -> u32 {
        self.get(w) >> shift & ((1 << bits) - 1)
    }

    fn submit(&self, kind: u32, words: &[u32]) {
        // SAFETY: a live object and the words of the command.
        assert_eq!(unsafe { ((*self.hal).vtbl.submit_command)(self.hal, kind, 1, 0, words.as_ptr() as *const c_void, 0) }, S_OK);
    }

    fn writable(&self) -> bool {
        self.field(1, 31, 1) == 1
    }

    fn ended(&self) -> bool {
        self.field(2, 31, 1) == 1
    }

    /// Waits until the ring is full or the input ended.
    fn settle(&self) {
        let t = Instant::now();
        while self.writable() && !self.ended() {
            assert!(t.elapsed() < Duration::from_secs(5), "the engine did not fill the ring");
            std::thread::sleep(Duration::from_millis(1));
        }
        // Let a pass that is running finish.
        std::thread::sleep(Duration::from_millis(5));
    }

    fn ring(&self) -> Vec<i16> {
        // SAFETY: the ring.
        let bytes = unsafe { std::slice::from_raw_parts(self.output.0, (RING * 256) as usize) };
        bytes.as_chunks::<2>().0.iter().map(|p| i16::from_le_bytes(*p)).collect()
    }

    /// Takes the whole ring again and again until the input ends, returns the blocks taken.
    fn drain(&self) -> u64 {
        let mut taken = 0;
        for _ in 0..1000 {
            self.settle();
            if self.ended() && self.writable() {
                break;
            }
            self.submit(CMD_XMA_OUTPUT_CONSUMED, &[INDEX, RING]);
            taken += u64::from(RING);
        }
        taken
    }
}

impl Drop for Setup {
    fn drop(&mut self) {
        // SAFETY: the object and the allocations of this setup.
        unsafe {
            ((*self.hal).vtbl.release)(self.hal);
            ApuFree(self.input.0 as *mut c_void);
            ApuFree(self.output.0 as *mut c_void);
        }
        assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    }
}

fn tone(frames: usize) -> Vec<u8> {
    decant_xma::synth::tone_stream(&[Some((21, 1))], 48_000, 26, frames)
}

#[test]
fn an_enabled_context_fills_its_ring_and_moves_its_positions() {
    let _s = serial();
    let s = Setup::new(&tone(10));
    std::thread::sleep(Duration::from_millis(10));
    assert_eq!(s.field(0, 27, 5), 0, "nothing is written before the enable");
    assert!(s.ring().iter().all(|&v| v == 0x5555));
    s.submit(CMD_XMA_ENABLE, &[INDEX]);
    s.settle();
    // Four blocks written: the write block went round to the read block and the ring is full.
    assert!(!s.writable());
    assert_eq!(s.field(0, 27, 5), 0);
    assert_eq!(xma::stats(INDEX).unwrap().2, 4);
    // One frame of 512 samples fills the four blocks: the read position is past the first frame.
    assert!(s.field(2, 0, 26) > 32, "read position {}", s.field(2, 0, 26));
    assert!(s.ring().iter().any(|&v| v.abs() > 100), "the tone is in the ring");
    // Taking two blocks frees two, which the engine fills: the write block moves to 2.
    s.submit(CMD_XMA_OUTPUT_CONSUMED, &[INDEX, 2]);
    assert_eq!(s.field(9, 0, 5), 2);
    s.settle();
    assert_eq!(s.field(0, 27, 5), 2);
    assert!(!s.writable());
    assert_eq!(xma::stats(INDEX).unwrap().2, 6);
}

#[test]
fn the_end_of_the_input_sets_the_error_status() {
    let _s = serial();
    let s = Setup::new(&tone(10));
    s.submit(CMD_XMA_ENABLE, &[INDEX]);
    let taken = s.drain();
    // 10 frames of 512 samples are 40 blocks, all of them written.
    let (frames, bad, written, _) = xma::stats(INDEX).unwrap();
    assert_eq!((frames, bad, written), (10, 0, 40));
    assert!(taken >= 36, "{taken} blocks taken");
    assert_eq!(s.field(2, 26, 5), xma::ERROR_INPUT_ENDED);
    assert!(s.ended());
    assert_eq!(s.field(0, 20, 2), 0, "the input is marked used up");
    // The title gives the buffer again: the error clears and decoding goes on.
    s.set(0, s.get(0) | 1 << 20);
    s.submit(CMD_XMA_OUTPUT_CONSUMED, &[INDEX, RING]);
    s.settle();
    assert!(!s.ended());
    assert!(xma::stats(INDEX).unwrap().0 > 10);
    s.submit(CMD_XMA_DISABLE, &[INDEX]);
    assert!(!xma::is_enabled(INDEX));
}

/// The bit position of the last frame of `data`.
fn last_frame(data: &[u8]) -> u32 {
    let mut at = decant_xma::first_frame(data).unwrap();
    while let Some(n) = decant_xma::read_frame(data, at).unwrap().next {
        at = n;
    }
    at as u32
}

#[test]
fn a_loop_plays_the_frames_again() {
    let _s = serial();
    let data = tone(6);
    let s = Setup::new(&data);
    // Loop twice from the first frame to the last, the whole last frame kept (loop end subframe 3).
    s.set(0, s.get(0) | 2 << 12);
    s.set(1, s.get(1) | 3 << 12);
    s.set(3, 32);
    s.set(4, last_frame(&data));
    s.submit(CMD_XMA_ENABLE, &[INDEX]);
    s.drain();
    let (frames, _, written, _) = xma::stats(INDEX).unwrap();
    assert_eq!(frames, 18, "6 frames played three times");
    assert_eq!(written, 18 * 4);
    assert_eq!(s.field(0, 12, 8), 0, "the loop count went down to 0");
    assert!(s.ended());
}

#[test]
fn a_loop_end_subframe_cuts_the_last_frame() {
    let _s = serial();
    let data = tone(4);
    let s = Setup::new(&data);
    // One loop, the loop end frame cut after its first subframe (128 samples, one block).
    s.set(0, s.get(0) | 1 << 12);
    s.set(3, 32);
    s.set(4, last_frame(&data));
    s.submit(CMD_XMA_ENABLE, &[INDEX]);
    s.drain();
    // First pass: 3 frames and 128 samples, second pass 4 frames: 12 + 1 + 16 blocks.
    assert_eq!(xma::stats(INDEX).unwrap().2, 29);
}

#[test]
fn a_loop_for_ever_never_ends() {
    let _s = serial();
    let data = tone(3);
    let s = Setup::new(&data);
    s.set(0, s.get(0) | 255 << 12);
    s.set(1, s.get(1) | 3 << 12);
    s.set(3, 32);
    s.set(4, last_frame(&data));
    s.submit(CMD_XMA_ENABLE, &[INDEX]);
    for _ in 0..20 {
        s.settle();
        assert!(!s.ended());
        s.submit(CMD_XMA_OUTPUT_CONSUMED, &[INDEX, RING]);
    }
    assert_eq!(s.field(0, 12, 8), 255);
    assert!(xma::stats(INDEX).unwrap().0 >= 20);
}

/// The title's own start, as measured in a game run: 350 XMA contexts, a connection, then command 2 with id 0xF0F0F0F0,
/// frame 0xFFFFFFFF and one data word 0xC0 (the message kinds error 0x40 and disconnected 0x80). The command is acknowledged
/// once, no audio frame start message follows (the title did not ask for them), and no context is decoded.
#[test]
fn the_titles_start_sequence() {
    let _s = serial();
    let mut c = ShapeContexts { counts: [0, 0, 0, 0, 350, 0], ..Default::default() };
    // SAFETY: a writable structure, then a live object and its one data word.
    unsafe {
        assert_eq!(AcpHalAllocateShapeContexts(&mut c), S_OK);
        let mut hal = std::ptr::null_mut();
        assert_eq!(AcpHalCreate(&mut hal), S_OK);
        let vt = (*hal).vtbl;
        assert_eq!((vt.connect)(hal, 0x40, 0x40), S_OK);
        let mask = 0xC0u32;
        assert_eq!((vt.submit_command)(hal, CMD_CONFIGURE, 0xF0F0_F0F0, 0xFFFF_FFFF, (&raw const mask).cast(), 0), S_OK);
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!((vt.get_num_messages)(hal), 1);
        let mut m = [0u8; 28];
        assert_eq!((vt.pop_message)(hal, m.as_mut_ptr()), 1);
        assert_eq!(u32::from_le_bytes(m[0..4].try_into().unwrap()), MSG_COMMAND_COMPLETED);
        assert_eq!(u64::from_le_bytes(m[16..24].try_into().unwrap()), 0xF0F0_F0F0);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!((vt.pop_message)(hal, m.as_mut_ptr()), 0, "no frame start message for mask 0xC0");
        assert!((0..350).all(|i| !xma::is_enabled(i)));
        (vt.release)(hal);
        assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    }
}
