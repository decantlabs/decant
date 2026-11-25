//! `AcpHalAllocateShapeContexts` and `AcpHalReleaseShapeContexts`, and the service thread.
#![allow(unsafe_code)]

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use AcpHal::abi::*;
use AcpHal::exports::*;
use AcpHal::shape::frame_counter;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // The heaps with their default sizes, whichever test runs first.
    let _ = ApuCreateHeap(0x400_0000, 0);
    guard
}

fn contexts(counts: [u32; 6]) -> ShapeContexts {
    ShapeContexts { counts, ..Default::default() }
}

fn alloc(c: &mut ShapeContexts) -> Hresult {
    // SAFETY: a writable structure of 0x60 bytes.
    unsafe { AcpHalAllocateShapeContexts(c) }
}

fn heap2() -> ApuHeapState {
    let mut s = ApuHeapState::default();
    // SAFETY: a writable structure.
    assert_eq!(unsafe { ApuHeapGetState(&mut s, 2) }, S_OK);
    s
}

#[test]
fn the_structure_is_0x60_bytes() {
    assert_eq!(std::mem::size_of::<ShapeContexts>(), 0x60);
    assert_eq!(std::mem::offset_of!(ShapeContexts, pointers), 0x18);
    assert_eq!(std::mem::offset_of!(ShapeContexts, apu_addresses), 0x48);
}

#[test]
fn invalid_counts_are_refused() {
    let _s = serial();
    // SAFETY: a null pointer is part of the contract.
    assert_eq!(unsafe { AcpHalAllocateShapeContexts(std::ptr::null_mut()) }, E_INVALIDARG);
    // Each count above its limit, with a consistent SRC count where it is checked.
    let limits = [0x1000u32, 0x1000, 0x1000, 0x1000, 0x200, 0xE00];
    for (i, limit) in limits.iter().enumerate() {
        let mut counts = [0u32; 6];
        counts[i] = limit + 1;
        assert_eq!(alloc(&mut contexts(counts)), E_INVALIDARG, "count {i}");
    }
    // SRC count must be XMA plus PCM when SRC or PCM is nonzero.
    assert_eq!(alloc(&mut contexts([3, 0, 0, 0, 1, 1])), E_INVALIDARG);
    assert_eq!(alloc(&mut contexts([0, 0, 0, 0, 1, 1])), E_INVALIDARG);
    assert_eq!(alloc(&mut contexts([2, 0, 0, 0, 0, 0])), E_INVALIDARG);
    assert!(!AcpHal::shape::is_allocated());
}

#[test]
fn contexts_are_allocated_in_heap_2_and_released() {
    let _s = serial();
    let before = heap2();
    let mut c = contexts([3, 2, 1, 4, 2, 1]);
    assert_eq!(alloc(&mut c), S_OK);
    assert!(AcpHal::shape::is_allocated());
    // Five allocations: the XMA and PCM contexts share one.
    assert_eq!(heap2().allocations, before.allocations + 5);
    for i in 0..6 {
        assert_ne!(c.pointers[i], 0, "array {i}");
        assert_ne!(c.apu_addresses[i], 0);
        assert_eq!(ApuMapApuAddress(c.apu_addresses[i]) as u64, c.pointers[i]);
        assert_eq!(ApuMapVirtualAddress(c.pointers[i] as *const _), c.apu_addresses[i]);
    }
    assert_eq!(c.pointers[0] % 0x20, 0);
    assert_eq!(c.pointers[1] % 0x10, 0);
    assert_eq!(c.pointers[4] % 0x40, 0);
    // PCM follows XMA in one allocation.
    assert_eq!(c.pointers[5], c.pointers[4] + 2 * 0x40);
    assert_eq!(c.apu_addresses[5], c.apu_addresses[4] + 2 * 0x40);
    // The contexts are zeroed.
    // SAFETY: the arrays hold 3 SRC contexts of 0x20 bytes.
    assert!(unsafe { std::slice::from_raw_parts(c.pointers[0] as *const u8, 0x60) }.iter().all(|b| *b == 0));

    assert_eq!(alloc(&mut contexts([0; 6])), SHAPE_E_ALREADY_ALLOCATED);
    assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    assert!(!AcpHal::shape::is_allocated());
    assert_eq!(heap2(), before);
    assert_eq!(AcpHalReleaseShapeContexts(), SHAPE_E_NOT_ALLOCATED);
}

#[test]
fn a_count_of_zero_gives_a_null_pointer_and_address() {
    let _s = serial();
    let mut c = contexts([0, 0, 5, 0, 0, 0]);
    c.pointers = [9; 6];
    c.apu_addresses = [9; 6];
    assert_eq!(alloc(&mut c), S_OK);
    for i in [0, 1, 3, 4, 5] {
        assert_eq!((c.pointers[i], c.apu_addresses[i]), (0, 0), "array {i}");
    }
    assert_ne!(c.pointers[2], 0);
    // Only XMA contexts: PCM stays empty.
    assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    let mut c = contexts([0, 0, 0, 0, 2, 0]);
    assert_eq!(alloc(&mut c), S_OK);
    assert_ne!(c.pointers[4], 0);
    assert_eq!(c.pointers[5], 0);
    assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
}

#[test]
fn the_maximum_contexts_fit_the_default_heap_2_exactly() {
    let _s = serial();
    let before = heap2();
    let mut c = contexts([0x1000, 0x1000, 0x1000, 0x1000, 0x200, 0xE00]);
    assert_eq!(alloc(&mut c), S_OK);
    assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    assert_eq!(heap2(), before);
}

#[test]
fn a_failed_allocation_frees_what_it_took() {
    let _s = serial();
    let before = heap2();
    // Leave 0x6000 bytes of heap 2: the first array fits and the second does not.
    let mut va = std::ptr::null_mut();
    let mut apu = 0;
    // SAFETY: writable out values.
    assert_eq!(unsafe { ApuAlloc(&mut va, &mut apu, 0x11_0000, 256, 2) }, S_OK);
    assert_eq!(alloc(&mut contexts([0x100, 0x100, 0, 0, 0x80, 0x80])), E_OUTOFMEMORY);
    assert_eq!(heap2().allocations, before.allocations + 1);
    assert!(!AcpHal::shape::is_allocated());
    assert_eq!(ApuFree(va), S_OK);
    assert_eq!(heap2(), before);
}

#[test]
fn the_service_thread_counts_audio_frames_until_the_release() {
    let _s = serial();
    let mut c = contexts([1, 1, 1, 1, 1, 0]);
    assert_eq!(alloc(&mut c), S_OK);
    let start = frame_counter();
    let t = Instant::now();
    while frame_counter() < start + 20 && t.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(frame_counter() >= start + 20, "375 audio frames per second");
    assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    let stopped = frame_counter();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(frame_counter(), stopped, "the engine stops with the release");
}
