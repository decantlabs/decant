//! The `IAcpHal` object, driven through its table of 12 functions as a C++ client does.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard};

use AcpHal::abi::*;
use AcpHal::exports::*;
use AcpHal::hal::{IAcpHal, IAcpHalVtbl};
use AcpHal::shape::engine_snapshot;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = ApuCreateHeap(0x400_0000, 0);
    guard
}

struct Hal(*mut IAcpHal);

impl Hal {
    fn new() -> Hal {
        let mut p = std::ptr::null_mut();
        // SAFETY: a writable out pointer.
        assert_eq!(unsafe { AcpHalCreate(&mut p) }, S_OK);
        assert!(!p.is_null());
        Hal(p)
    }
    fn vt(&self) -> &'static IAcpHalVtbl {
        // SAFETY: the object is live.
        unsafe { (*self.0).vtbl }
    }
    fn obj(&self) -> &IAcpHal {
        // SAFETY: the object is live.
        unsafe { &*self.0 }
    }
    fn connect(&self, commands: u32, messages: u32) -> Hresult {
        // SAFETY: a live object.
        unsafe { (self.vt().connect)(self.0, commands, messages) }
    }
    fn disconnect(&self) -> Hresult {
        // SAFETY: a live object.
        unsafe { (self.vt().disconnect)(self.0) }
    }
    fn submit(&self, kind: u32, id: u64, payload: &[u32], fence: u32) -> Hresult {
        let p = if payload.is_empty() { std::ptr::null() } else { payload.as_ptr() as *const c_void };
        // SAFETY: a live object and a payload of the words the command reads.
        unsafe { (self.vt().submit_command)(self.0, kind, id, 7, p, fence) }
    }
    fn count(&self) -> u32 {
        // SAFETY: a live object.
        unsafe { (self.vt().get_num_messages)(self.0) }
    }
    fn pop(&self) -> Option<[u8; 28]> {
        let mut out = [0xEEu8; 28];
        // SAFETY: a live object and a buffer of 28 bytes.
        (unsafe { (self.vt().pop_message)(self.0, out.as_mut_ptr()) } == 1).then_some(out)
    }
    fn release(self) {
        // SAFETY: gives up the only reference.
        assert_eq!(unsafe { (self.vt().release)(self.0) }, 0);
    }
}

fn u32_at(m: &[u8; 28], off: usize) -> u32 {
    u32::from_le_bytes(m[off..off + 4].try_into().unwrap())
}

fn u64_at(m: &[u8; 28], off: usize) -> u64 {
    u64::from_le_bytes(m[off..off + 8].try_into().unwrap())
}

fn heap1_allocations() -> u32 {
    let mut s = ApuHeapState::default();
    // SAFETY: a writable structure.
    assert_eq!(unsafe { ApuHeapGetState(&mut s, 1) }, S_OK);
    s.allocations
}

/// A word of APU memory: (host pointer, APU address).
fn apu_words(bytes: u32) -> (*mut u8, u32) {
    let mut va = std::ptr::null_mut();
    let mut apu = 0;
    // SAFETY: writable out values.
    assert_eq!(unsafe { ApuAlloc(&mut va, &mut apu, bytes, 256, 2) }, S_OK);
    (va as *mut u8, apu)
}

#[test]
fn the_table_has_12_slots() {
    assert_eq!(std::mem::size_of::<IAcpHalVtbl>(), 12 * 8);
    assert_eq!(std::mem::offset_of!(IAcpHalVtbl, release), 5 * 8);
    assert_eq!(std::mem::offset_of!(IAcpHalVtbl, unsupported), 6 * 8);
    assert_eq!(std::mem::offset_of!(IAcpHalVtbl, map_driver_region), 10 * 8);
    assert_eq!(std::mem::offset_of!(IAcpHalVtbl, destructor), 11 * 8);
    assert_eq!(std::mem::offset_of!(IAcpHal, vtbl), 0);
    // SAFETY: a null out pointer is part of the contract.
    assert_eq!(unsafe { AcpHalCreate(std::ptr::null_mut()) }, E_INVALIDARG);
}

#[test]
fn objects_have_unique_ids() {
    let _s = serial();
    let (a, b) = (Hal::new(), Hal::new());
    assert_ne!(a.obj().id(), b.obj().id());
    a.release();
    b.release();
}

#[test]
fn connect_and_disconnect() {
    let _s = serial();
    let before = heap1_allocations();
    let h = Hal::new();
    assert_eq!(h.submit(CMD_CONFIGURE, 1, &[2], 0), HAL_S_PENDING, "not connected");
    assert_eq!(h.disconnect(), HAL_E_NOT_CONNECTED);
    assert_eq!(h.connect(0, 8), E_INVALIDARG);
    assert!(!h.obj().is_connected());
    assert_eq!(h.connect(4, 8), S_OK);
    assert_eq!(h.connect(4, 8), HAL_E_ALREADY_CONNECTED);
    assert_eq!(h.connect(0, 8), HAL_E_ALREADY_CONNECTED, "already connected comes first");
    assert_eq!(heap1_allocations(), before + 2, "a command ring and a message ring");
    let (cmd, msg) = h.obj().ring_addresses();
    assert_ne!(ApuMapApuAddress(cmd.unwrap()), std::ptr::null_mut());
    assert_ne!(ApuMapApuAddress(msg.unwrap()), std::ptr::null_mut());
    assert_eq!(cmd.unwrap() % 0x100, 0);
    assert_eq!(h.count(), 0);
    assert_eq!(h.disconnect(), S_OK);
    assert_eq!(h.disconnect(), HAL_E_NOT_CONNECTED);
    // The rings stay allocated, a disconnected message is queued.
    assert_eq!(heap1_allocations(), before + 2);
    assert_eq!(h.count(), 1);
    let m = h.pop().unwrap();
    assert_eq!(u32_at(&m, 0), MSG_DISCONNECTED);
    assert_eq!(u32_at(&m, 4), 0);
    assert_eq!(u32_at(&m, 12), h.obj().id());
    // Reconnecting frees the old rings and takes new ones.
    assert_eq!(h.connect(2, 0), S_OK);
    assert_eq!(heap1_allocations(), before + 2);
    h.release();
    assert_eq!(heap1_allocations(), before, "the release frees the rings");
}

#[test]
fn at_most_four_objects_are_connected() {
    let _s = serial();
    let hals: Vec<Hal> = (0..5).map(|_| Hal::new()).collect();
    for h in &hals[..4] {
        assert_eq!(h.connect(1, 1), S_OK);
    }
    assert_eq!(hals[4].connect(1, 1), E_UNEXPECTED);
    assert_eq!(hals[0].disconnect(), S_OK);
    assert_eq!(hals[4].connect(1, 1), S_OK);
    assert_eq!(hals[0].connect(1, 1), E_UNEXPECTED);
    for h in hals {
        h.release();
    }
    // The count goes down with the objects.
    let h = Hal::new();
    assert_eq!(h.connect(1, 1), S_OK);
    h.release();
}

#[test]
fn submit_validates_type_and_payload() {
    let _s = serial();
    let h = Hal::new();
    assert_eq!(h.connect(4, 16), S_OK);
    assert_eq!(h.submit(0, 1, &[0], 0), E_INVALIDARG);
    assert_eq!(h.submit(0x1C, 1, &[0], 0), E_INVALIDARG);
    assert_eq!(h.submit(0xFFFF, 1, &[0], 0), E_INVALIDARG);
    assert_eq!(h.submit(CMD_XMA_ENABLE, 1, &[], 0), E_INVALIDARG, "null payload");
    assert_eq!(h.submit(CMD_RUN_FLOWGRAPH, 1, &[], 0), E_INVALIDARG, "null payload");
    assert_eq!(h.submit(4, 1, &[], 0), E_NOTIMPL, "type 4 may have no payload");
    assert_eq!(h.submit(3, 1, &[0], 0), E_NOTIMPL);
    assert_eq!(h.submit(0x1B, 1, &[0], 0), E_NOTIMPL);
    assert_eq!(h.submit(0x11, 1, &[0], 0), E_NOTIMPL);
    assert_eq!(h.count(), 0, "a refused command leaves no message");
    h.release();
}

#[test]
fn a_command_completes_with_a_message_and_a_fence() {
    let _s = serial();
    // The audio clock counts since the process started and stands still while no service thread runs,
    // so it keeps what an earlier test left. Compare with the values read before the command.
    let clock_before = AcpHal::shape::audio_clock();
    let frames_before = AcpHal::shape::frame_counter() as u32;
    let h = Hal::new();
    assert_eq!(h.connect(4, 16), S_OK);
    let (fence_ptr, fence) = apu_words(16);
    assert_eq!(h.submit(CMD_CONFIGURE, 0x1122_3344_5566, &[0xABCE], fence), S_OK);
    assert_eq!(engine_snapshot().config, 0xABCE);
    // SAFETY: the fence word in APU memory.
    assert_eq!(unsafe { *(fence_ptr as *const u32) }, 1);
    assert_eq!(h.count(), 1);
    let m = h.pop().unwrap();
    assert_eq!(u32_at(&m, 0), MSG_COMMAND_COMPLETED);
    assert_eq!(u32_at(&m, 4), 0);
    assert_eq!(u32_at(&m, 8), clock_before, "the audio clock does not move while the engine does not run");
    assert_eq!(u32_at(&m, 12), h.obj().id());
    assert_eq!(u64_at(&m, 16), 0x1122_3344_5566);
    assert_eq!(u32_at(&m, 24), frames_before, "no audio frame is counted while the engine does not run");
    // A command id of 0 queues no message. A fence that does not map is ignored.
    assert_eq!(h.submit(CMD_CONFIGURE, 0, &[6], 0x7777_7777), S_OK);
    assert_eq!(h.count(), 0);
    assert_eq!(engine_snapshot().config, 6);
    assert_eq!(h.obj().commands_run(), 2);
    assert!(h.pop().is_none());
    assert_eq!(ApuFree(fence_ptr as *mut c_void), S_OK);
    h.release();
}

#[test]
fn messages_come_out_in_order_and_the_ring_drops_when_full() {
    let _s = serial();
    let h = Hal::new();
    assert_eq!(h.connect(2, 3), S_OK);
    for id in 1..=5u64 {
        assert_eq!(h.submit(CMD_CONFIGURE, id, &[id as u32 * 2], 0), S_OK);
    }
    assert_eq!(h.count(), 3, "capacity 3, the later messages are dropped");
    // SAFETY: a null buffer is part of the contract.
    assert_eq!(unsafe { (h.vt().pop_message)(h.0, std::ptr::null_mut()) }, 0);
    assert_eq!(h.count(), 3, "a null buffer pops nothing");
    for id in 1..=3u64 {
        assert_eq!(u64_at(&h.pop().unwrap(), 16), id);
    }
    assert!(h.pop().is_none());
    assert_eq!(h.submit(CMD_CONFIGURE, 9, &[2], 0), S_OK);
    assert_eq!(u64_at(&h.pop().unwrap(), 16), 9);
    h.release();
    // A message count of 0 means 256.
    let h = Hal::new();
    assert_eq!(h.connect(1, 0), S_OK);
    for _ in 0..300 {
        assert_eq!(h.submit(CMD_CONFIGURE, 1, &[2], 0), S_OK);
    }
    assert_eq!(h.count(), 256);
    h.release();
}

#[test]
fn commands_are_written_to_the_ring_slots_in_turn() {
    let _s = serial();
    let h = Hal::new();
    assert_eq!(h.connect(2, 8), S_OK);
    let ring = ApuMapApuAddress(h.obj().ring_addresses().0.unwrap()) as *const u8;
    let slot = |n: usize| {
        // SAFETY: the ring holds 2 slots of 0x10C bytes.
        let s = unsafe { std::slice::from_raw_parts(ring.add(n * 0x10C), 0x10C) };
        (u32::from_le_bytes(s[0..4].try_into().unwrap()), u64::from_le_bytes(s[8..16].try_into().unwrap()), u32::from_le_bytes(s[16..20].try_into().unwrap()), u32::from_le_bytes(s[24..28].try_into().unwrap()))
    };
    assert_eq!(h.submit(CMD_CONFIGURE, 11, &[100], 0), S_OK);
    assert_eq!(h.submit(CMD_XMA_ENABLE, 12, &[3], 0), S_OK);
    assert_eq!(slot(0), (CMD_CONFIGURE, 11, 7, 100));
    assert_eq!(slot(1), (CMD_XMA_ENABLE, 12, 7, 3));
    assert_eq!(h.submit(CMD_CONFIGURE, 13, &[300], 0), S_OK);
    assert_eq!(slot(0), (CMD_CONFIGURE, 13, 7, 300), "the third command wraps to slot 0");
    h.release();
}

#[test]
fn xma_context_commands_change_the_engine_state() {
    let _s = serial();
    let h = Hal::new();
    assert_eq!(h.connect(4, 32), S_OK);
    let mask = |i: u32| engine_snapshot().xma_enabled[(i / 32) as usize] >> (i % 32) & 1;
    assert_eq!(h.submit(CMD_XMA_DISABLE_MASK, 0, &[u32::MAX; 16], 0), S_OK);
    assert_eq!(engine_snapshot().xma_enabled, [0; 16]);
    assert_eq!(h.submit(CMD_XMA_ENABLE, 0, &[0x1FF], 0), S_OK);
    assert_eq!(mask(0x1FF), 1);
    assert_eq!(h.submit(CMD_XMA_ENABLE, 0, &[0x200], 0), E_INVALIDARG);
    assert_eq!(h.submit(CMD_XMA_DISABLE, 0, &[0x200], 0), E_INVALIDARG);
    let mut words = [0u32; 16];
    words[0] = 0b1010;
    words[15] = 1 << 31;
    assert_eq!(h.submit(CMD_XMA_ENABLE_MASK, 0, &words, 0), S_OK);
    assert_eq!((mask(1), mask(2), mask(3), mask(511), mask(0x1FF)), (1, 0, 1, 1, 1));
    assert_eq!(h.submit(CMD_XMA_DISABLE, 0, &[3], 0), S_OK);
    assert_eq!(mask(3), 0);
    assert_eq!(h.submit(CMD_XMA_DISABLE_MASK, 0, &words, 0), S_OK);
    assert_eq!((mask(1), mask(2), mask(3), mask(511)), (0, 0, 0, 0));
    assert_eq!(h.submit(CMD_XMA_UPDATE, 0, &[5, 0x99, 0x1234_5600], 0), S_OK);
    assert_eq!(engine_snapshot().xma_update[&5], (0x99, 0x1234_5600));
    assert_eq!(h.submit(CMD_XMA_UPDATE, 0, &[0x200, 0, 0], 0), E_INVALIDARG);
    assert_eq!(h.submit(CMD_XMA_SET_INPUT, 0, &[9], 0), S_OK);
    assert!(engine_snapshot().xma_input.contains_key(&9));
    let before = engine_snapshot().xma_output_consumed.get(&4).copied().unwrap_or(0);
    assert_eq!(h.submit(CMD_XMA_OUTPUT_CONSUMED, 0, &[4, 6], 0), S_OK);
    assert_eq!(h.submit(CMD_XMA_OUTPUT_CONSUMED, 0, &[4, 2], 0), S_OK);
    assert_eq!(engine_snapshot().xma_output_consumed[&4], before + 8);
    assert_eq!(h.submit(CMD_XMA_OUTPUT_CONSUMED, 0, &[0x200, 2], 0), E_INVALIDARG);
    let before = engine_snapshot().dma_consumed.get(&0xFFF).copied().unwrap_or(0);
    assert_eq!(h.submit(CMD_DMA_CONSUMED, 0, &[0xFFF, 3], 0), S_OK);
    assert_eq!(engine_snapshot().dma_consumed[&0xFFF], before + 3);
    assert_eq!(h.submit(CMD_DMA_CONSUMED, 0, &[0x1000, 3], 0), E_INVALIDARG);
    h.release();
}

#[test]
fn a_flowgraph_completes_or_terminates() {
    let _s = serial();
    let h = Hal::new();
    assert_eq!(h.connect(4, 16), S_OK);
    let (ptr, apu) = apu_words(0x100);
    let ops: [u64; 3] = [1 | (5 << 33), 2 | (3 << 13), 0x10];
    // SAFETY: 0x100 bytes of APU memory.
    unsafe { std::ptr::copy_nonoverlapping(ops.as_ptr(), ptr as *mut u64, 3) };
    let before = engine_snapshot();
    assert_eq!(h.submit(CMD_RUN_FLOWGRAPH, 0x77, &[3, apu], 0), S_OK);
    let m = h.pop().unwrap();
    assert_eq!(u32_at(&m, 0), MSG_FLOWGRAPH_COMPLETED);
    assert_eq!(u32_at(&m, 12), apu);
    assert_eq!(u64_at(&m, 16), 0x77);
    assert!(h.pop().is_none(), "no extra command completed message");
    let after = engine_snapshot();
    assert!(after.mix_buffers.contains_key(&5));
    assert_eq!(after.ops_clear, before.ops_clear + 1);
    assert_eq!(after.ops_unimplemented, before.ops_unimplemented + 2);
    // The completed message is queued whatever the command id is.
    assert_eq!(h.submit(CMD_RUN_FLOWGRAPH, 0, &[1, apu], 0), S_OK);
    assert_eq!(u32_at(&h.pop().unwrap(), 0), MSG_FLOWGRAPH_COMPLETED);
    // A count of 0, an address that does not map and a graph that leaves its allocation terminate.
    for words in [[0, apu], [3, 0x1234], [3, 0], [0x1000, apu]] {
        assert_eq!(h.submit(CMD_RUN_FLOWGRAPH, 0x78, &words, 0), S_OK);
        let m = h.pop().unwrap();
        assert_eq!(u32_at(&m, 0), MSG_FLOWGRAPH_TERMINATED, "{words:x?}");
        assert_eq!(u32_at(&m, 12), words[1]);
        assert_eq!(u64_at(&m, 16), 0x78);
    }
    assert_eq!(ApuFree(ptr as *mut c_void), S_OK);
    h.release();
}

#[test]
fn the_other_slots_of_the_table() {
    let _s = serial();
    let h = Hal::new();
    for slot in &h.vt().unsupported {
        // SAFETY: a live object.
        assert_eq!(unsafe { slot(h.0) }, HAL_E_UNSUPPORTED);
    }
    // SAFETY: a null out pointer is part of the contract.
    assert_eq!(unsafe { (h.vt().map_driver_region)(h.0, std::ptr::null_mut()) }, E_INVALIDARG);
    let mut out = 5u64;
    // SAFETY: a writable UINT64.
    assert_eq!(unsafe { (h.vt().map_driver_region)(h.0, &mut out) }, 0x8007_0032_u32 as i32);
    assert_eq!(out, 0);

    // The destructor without the free flag disconnects, frees the rings and clears the messages.
    let before = heap1_allocations();
    assert_eq!(h.connect(2, 4), S_OK);
    assert_eq!(h.submit(CMD_CONFIGURE, 1, &[2], 0), S_OK);
    // SAFETY: a live object.
    assert_eq!(unsafe { (h.vt().destructor)(h.0, 0) }, h.0);
    assert!(!h.obj().is_connected());
    assert_eq!(h.count(), 0);
    assert_eq!(heap1_allocations(), before);
    assert_eq!(h.connect(2, 4), S_OK, "the object can be used again");
    // With the free flag it frees the object and returns null.
    // SAFETY: the object is not used afterwards.
    assert!(unsafe { (h.vt().destructor)(h.0, 1) }.is_null());
    assert_eq!(heap1_allocations(), before);
}

#[test]
fn messages_carry_the_audio_clock_while_the_engine_runs() {
    let _s = serial();
    let mut c = ShapeContexts { counts: [1, 0, 0, 0, 1, 0], ..Default::default() };
    // SAFETY: a writable structure.
    assert_eq!(unsafe { AcpHalAllocateShapeContexts(&mut c) }, S_OK);
    let h = Hal::new();
    assert_eq!(h.connect(1, 8), S_OK);
    let t = std::time::Instant::now();
    loop {
        assert_eq!(h.submit(CMD_CONFIGURE, 1, &[2], 0), S_OK);
        let m = h.pop().unwrap();
        if u32_at(&m, 24) > 0 {
            assert_eq!(u32_at(&m, 8) % 128, 0, "the clock counts whole audio frames of 128 samples");
            assert!(i64::from(u32_at(&m, 8)) >= i64::from(u32_at(&m, 24)) * 128 - 128 * 8);
            break;
        }
        assert!(t.elapsed().as_secs() < 5, "the frame counter never moved");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    h.release();
    assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
}

/// Pops every message, returns them in order.
fn drain(h: &Hal) -> Vec<[u8; 28]> {
    std::iter::from_fn(|| h.pop()).collect()
}

#[test]
fn command_2_without_data_is_acknowledged() {
    let _s = serial();
    let h = Hal::new();
    assert_eq!(h.connect(16, 64), S_OK);
    let (fence_ptr, fence) = apu_words(16);
    // As the title sends it right after Connect: type 2, id 0xF0F0F0F0, no data.
    assert_eq!(h.submit(CMD_CONFIGURE, 0xF0F0_F0F0, &[], fence), S_OK);
    // SAFETY: the fence word in APU memory.
    assert_eq!(unsafe { *(fence_ptr as *const u32) }, 1);
    let m = h.pop().unwrap();
    assert_eq!(u32_at(&m, 0), MSG_COMMAND_COMPLETED);
    assert_eq!(u32_at(&m, 12), h.obj().id());
    assert_eq!(u64_at(&m, 16), 0xF0F0_F0F0);
    assert_eq!(h.obj().commands_run(), 1);
    assert_eq!(ApuFree(fence_ptr as *mut c_void), S_OK);
    h.release();
}

#[test]
fn frame_messages_arrive_every_128_samples() {
    let _s = serial();
    let h = Hal::new();
    assert_eq!(h.connect(4, 256), S_OK);
    let start = std::time::Instant::now();
    assert_eq!(h.submit(CMD_CONFIGURE, 0, &[], 0), S_OK);
    let mut frames = Vec::new();
    while start.elapsed() < std::time::Duration::from_millis(400) {
        for m in drain(&h) {
            assert_eq!(u32_at(&m, 0), MSG_AUDIO_FRAME_START);
            frames.push((u32_at(&m, 8), u32_at(&m, 12), u32_at(&m, 24)));
        }
        std::thread::sleep(std::time::Duration::from_millis(3));
    }
    let elapsed = start.elapsed().as_secs_f64();
    // 375 frames per second. Read every few milliseconds, so none is skipped: the numbers run 0, 1, 2 ...
    let expected = elapsed * 375.0;
    assert!((frames.len() as f64) <= expected + 1.0, "{} frames in {elapsed} s", frames.len());
    assert!((frames.len() as f64) >= expected - 12.0, "{} frames in {elapsed} s", frames.len());
    for (n, (clock, connection, frame)) in frames.iter().enumerate() {
        assert_eq!(*frame, n as u32);
        assert_eq!(*clock, n as u32 * 128);
        assert_eq!(*connection, h.obj().id());
    }
    h.release();
}

#[test]
fn a_late_reader_gets_at_most_8_frames() {
    let _s = serial();
    let h = Hal::new();
    assert_eq!(h.connect(4, 256), S_OK);
    assert_eq!(h.submit(CMD_CONFIGURE, 0, &[], 0), S_OK);
    std::thread::sleep(std::time::Duration::from_millis(100));
    let frames = drain(&h);
    assert_eq!(frames.len() as u64, AcpHal::hal::MAX_LATE_FRAMES);
    let first = u32_at(&frames[0], 24);
    assert!(first >= 25, "the old frames are skipped, the first is {first}");
    for (n, m) in frames.iter().enumerate() {
        assert_eq!(u32_at(m, 24), first + n as u32, "the kept frames are consecutive");
    }
    h.release();
}

#[test]
fn frame_messages_follow_the_registered_mask() {
    let _s = serial();
    // Before any command 2 nothing is registered.
    let h = Hal::new();
    assert_eq!(h.connect(4, 256), S_OK);
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(h.count(), 0);
    // A mask without the frame start bit (every other kind): no frame messages.
    assert_eq!(h.submit(CMD_CONFIGURE, 0, &[0xFE], 0), S_OK);
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(h.count(), 0);
    // With the bit they start.
    assert_eq!(h.submit(CMD_CONFIGURE, 0, &[MSG_AUDIO_FRAME_START], 0), S_OK);
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(h.count() > 0);
    assert_eq!(u32_at(&h.pop().unwrap(), 0), MSG_AUDIO_FRAME_START);
    h.release();
}
