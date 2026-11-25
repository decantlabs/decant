//! The heap exports, called in the order a title calls them. The state is process wide, so the whole life cycle is one test.
#![allow(unsafe_code)]

use std::ffi::c_void;

use AcpHal::abi::*;
use AcpHal::exports::*;

fn state(heap: u32) -> (Hresult, ApuHeapState) {
    let mut s = ApuHeapState { free_bytes: 7, used_bytes: 7, overhead_bytes: 7, largest_free_block: 7, allocations: 7 };
    // SAFETY: a writable structure.
    let r = unsafe { ApuHeapGetState(&mut s, heap) };
    (r, s)
}

fn alloc(size: u32, alignment: u32, heap: u32) -> (Hresult, *mut c_void, u32) {
    let mut va = std::ptr::dangling_mut::<c_void>();
    let mut apu = 0x55u32;
    // SAFETY: writable out values.
    let r = unsafe { ApuAlloc(&mut va, &mut apu, size, alignment, heap) };
    (r, va, apu)
}

#[test]
fn the_life_cycle_of_the_heaps() {
    // Before creation.
    assert_eq!(state(1).0, APU_E_HEAP_NOT_CREATED);
    assert_eq!(state(2).0, APU_E_HEAP_NOT_CREATED);
    assert_eq!(state(0).0, E_INVALIDARG);
    assert_eq!(state(3).0, E_INVALIDARG);
    // SAFETY: a null state is part of the contract.
    assert_eq!(unsafe { ApuHeapGetState(std::ptr::null_mut(), 1) }, E_INVALIDARG);
    let stray = 0u8;
    assert_eq!(ApuMapVirtualAddress(&stray as *const u8 as *const c_void), 0);
    assert!(ApuMapApuAddress(0x100_0000).is_null());
    assert_eq!(ApuIsVirtualAddressValid(&stray as *const u8 as *const c_void, 1), 0);
    assert_eq!(ApuFree(&stray as *const u8 as *mut c_void), E_INVALIDARG);

    // Creation.
    assert_eq!(ApuCreateHeap(0, 0), E_INVALIDARG);
    assert_eq!(state(1).0, APU_E_HEAP_NOT_CREATED, "a refused creation creates nothing");
    assert_eq!(ApuCreateHeap(0x10_0000, 0x2_0000), S_OK);
    assert_eq!(ApuCreateHeap(0x10_0000, 0x2_0000), APU_E_HEAP_ALREADY_CREATED);
    assert_eq!(ApuCreateHeap(0, 0), E_INVALIDARG, "the argument check comes first");

    let (r, s1) = state(1);
    assert_eq!(r, S_OK);
    assert_eq!(s1, ApuHeapState { free_bytes: 0x10_0000 - 0x101C, used_bytes: 0x101C, overhead_bytes: 0, largest_free_block: 0x10_0000 - 0x101C, allocations: 3 });
    let (_, s2) = state(2);
    assert_eq!(s2, ApuHeapState { free_bytes: 0x2_2000, used_bytes: 0, overhead_bytes: 0, largest_free_block: 0x2_2000, allocations: 0 });

    // Allocation errors clear the outputs.
    let (r, va, apu) = alloc(16, 256, 0);
    assert_eq!((r, va.is_null(), apu), (E_INVALIDARG, true, 0));
    for (size, alignment, heap) in [(0, 256, 1), (16, 0, 1), (16, 2, 1), (16, 6, 1), (16, 3, 2), (16, 256, 3)] {
        assert_eq!(alloc(size, alignment, heap).0, E_INVALIDARG, "{size} {alignment} {heap}");
    }
    let mut apu = 9u32;
    // SAFETY: a null virtual address pointer is part of the contract.
    assert_eq!(unsafe { ApuAlloc(std::ptr::null_mut(), &mut apu, 16, 256, 1) }, E_INVALIDARG);
    assert_eq!(apu, 0);
    assert_eq!(alloc(0x20_0000, 256, 1).0, E_OUTOFMEMORY);
    assert_eq!(state(1).1.allocations, 3);

    // A good allocation, without and with the APU address.
    let (r, va, apu) = alloc(0x180, 512, 1);
    assert_eq!(r, S_OK);
    assert!(!va.is_null());
    assert_eq!(apu % 512, 0);
    assert_eq!(va as usize % 512, 0);
    assert_eq!(state(1).1.allocations, 4);
    let mut va2 = std::ptr::null_mut();
    // SAFETY: a null APU address pointer is allowed.
    assert_eq!(unsafe { ApuAlloc(&mut va2, std::ptr::null_mut(), 0x40, 4, 2) }, S_OK);
    assert_eq!(state(2).1.allocations, 1);

    // The mappings.
    assert_eq!(ApuMapVirtualAddress(va), apu);
    assert_eq!(ApuMapVirtualAddress((va as usize + 0x17F) as *const c_void), apu + 0x17F);
    assert_eq!(ApuMapVirtualAddress(std::ptr::null()), 0);
    assert_eq!(ApuMapApuAddress(apu), va);
    assert_eq!(ApuMapApuAddress(apu + 0x10) as usize, va as usize + 0x10);
    assert!(ApuMapApuAddress(0).is_null());
    assert_eq!(ApuIsVirtualAddressValid(va, 0), 1);
    assert_eq!(ApuIsVirtualAddressValid(va, 512), 1);
    assert_eq!(ApuIsVirtualAddressValid((va as usize + 4) as *const c_void, 512), 0);
    assert_eq!(ApuIsVirtualAddressValid((va as usize + 4) as *const c_void, 4), 1);
    assert_eq!(ApuIsVirtualAddressValid(&stray as *const u8 as *const c_void, 1), 0);

    // Freeing.
    assert_eq!(ApuFree(std::ptr::null_mut()), E_INVALIDARG);
    assert_eq!(ApuFree((va as usize + 4) as *mut c_void), E_INVALIDARG);
    assert_eq!(ApuFree(va), S_OK);
    assert_eq!(ApuFree(va), E_INVALIDARG);
    assert_eq!(ApuMapVirtualAddress(va), 0);
    assert_eq!(state(1).1, s1);
    assert_eq!(ApuFree(va2), S_OK);
    assert_eq!(state(2).1, s2);
}
