//! The heaps appear with their default sizes at the first `ApuAlloc`.
#![allow(unsafe_code)]

use std::ffi::c_void;

use AcpHal::abi::*;
use AcpHal::exports::*;

#[test]
fn the_first_alloc_creates_the_default_heaps() {
    let mut va: *mut c_void = std::ptr::null_mut();
    let mut apu = 0u32;
    // SAFETY: writable out values.
    assert_eq!(unsafe { ApuAlloc(&mut va, &mut apu, 0x100, 256, 2) }, S_OK);
    assert!(apu >= 0x500_0000, "heap 2 starts at 0x5000000 with the default heap 1");
    let mut s = ApuHeapState::default();
    // SAFETY: a writable structure.
    assert_eq!(unsafe { ApuHeapGetState(&mut s, 1) }, S_OK);
    assert_eq!(s.allocations, 3);
    assert_eq!(s.free_bytes + s.used_bytes + s.overhead_bytes, 0x400_0000);
    // SAFETY: as above.
    assert_eq!(unsafe { ApuHeapGetState(&mut s, 2) }, S_OK);
    assert_eq!(s.allocations, 1);
    assert_eq!(s.free_bytes + s.used_bytes + s.overhead_bytes, 0x11_4000 + 0x2000);
    assert_eq!(ApuCreateHeap(0x100000, 0), APU_E_HEAP_ALREADY_CREATED);
}
