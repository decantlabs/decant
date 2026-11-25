use super::*;

fn small() -> Heaps {
    Heaps::new(0x20000, 0x8000).unwrap()
}

#[test]
fn a_fresh_heap_holds_the_three_blocks_of_the_library() {
    let h = small();
    let s = h.state(1).unwrap();
    assert_eq!(s.allocations, 3);
    assert_eq!(s.used_bytes, 0x1000 + 0x14 + 8);
    assert_eq!(s.overhead_bytes, 0);
    assert_eq!(s.free_bytes, 0x20000 - 0x101C);
    assert_eq!(s.largest_free_block, 0x20000 - 0x101C);
    let s2 = h.state(2).unwrap();
    assert_eq!(s2.allocations, 0);
    assert_eq!(s2.free_bytes, 0x8000 + 0x2000);
    assert_eq!(s2.largest_free_block, 0xA000);
    // The internal heap holds the 0x4C byte block.
    assert_eq!(h.heaps[2].live.len(), 1);
    assert_eq!(h.heaps[2].live.values().next().unwrap().requested, 0x4C);
}

#[test]
fn heap_addresses_follow_the_specification() {
    let h = Heaps::new(DEFAULT_HEAP1_SIZE, 0).unwrap();
    assert_eq!(h.extent(0), (0x100_0000, 0x400_0000));
    assert_eq!(h.extent(1), (0x500_0000, 0x116000));
    // Heap 2 ends at 0x5116000, the internal heap starts at the next multiple of 64 KiB.
    assert_eq!(h.extent(2), (0x512_0000, 0x1000));
    for heap in &h.heaps {
        assert_eq!(heap.mem as usize % HEAP_HOST_ALIGN, 0);
    }
}

#[test]
fn sizes_are_rounded_up_to_4_kib() {
    let h = Heaps::new(0x20001, 1).unwrap();
    assert_eq!(h.extent(0).1, 0x21000);
    assert_eq!(h.extent(1).1, 0x3000);
}

#[test]
fn a_heap_too_small_for_the_library_blocks_is_out_of_memory() {
    assert_eq!(Heaps::new(0x100, 0).err(), Some(E_OUTOFMEMORY));
    assert_eq!(Heaps::new(0xFFFF_F000, 0).err(), Some(E_OUTOFMEMORY));
}

#[test]
fn alloc_validates_its_arguments() {
    let mut h = small();
    for (size, alignment, heap) in [(0, 256, 1), (16, 0, 1), (16, 3, 1), (16, 2, 1), (16, 1, 1), (16, 6, 1), (16, 256, 0), (16, 256, 3), (16, 256, 99)] {
        assert_eq!(h.alloc(size, alignment, heap).err(), Some(E_INVALIDARG), "{size} {alignment} {heap}");
    }
    assert!(h.alloc(16, 4, 1).is_ok());
    assert!(h.alloc(16, 8, 2).is_ok());
}

#[test]
fn small_alignments_are_raised_to_256_and_the_size_rounds_up() {
    let mut h = small();
    let a = h.alloc(1, 4, 1).unwrap();
    assert_eq!(a.apu % 256, 0);
    assert_eq!(h.state(1).unwrap().overhead_bytes, 255);
    let b = h.alloc(300, 256, 1).unwrap();
    assert_eq!(b.apu % 256, 0);
    let s = h.state(1).unwrap();
    assert_eq!(s.overhead_bytes, 255 + 212);
    assert_eq!(s.used_bytes, 0x101C + 1 + 300);
    let c = h.alloc(8, 0x1000, 1).unwrap();
    assert_eq!(c.apu % 0x1000, 0);
    assert_eq!(c.host as usize % 0x1000, 0);
}

#[test]
fn the_allocation_is_zeroed_and_inside_the_heap() {
    let mut h = small();
    let a = h.alloc(0x400, 256, 2).unwrap();
    // SAFETY: 0x400 bytes just allocated.
    unsafe {
        assert!(std::slice::from_raw_parts(a.host, 0x400).iter().all(|b| *b == 0));
        std::ptr::write_bytes(a.host, 0xFF, 0x400);
    }
    assert!(a.apu >= h.extent(1).0 && a.apu < h.extent(1).0 + h.extent(1).1);
    h.free(a.host).unwrap();
    let b = h.alloc(0x400, 256, 2).unwrap();
    assert_eq!(a.host, b.host);
    // SAFETY: as above.
    assert!(unsafe { std::slice::from_raw_parts(b.host, 0x400) }.iter().all(|b| *b == 0));
}

#[test]
fn best_fit_takes_the_smallest_block_that_holds_the_request() {
    let mut h = small();
    let x1 = h.alloc(0x300, 256, 1).unwrap();
    let _x2 = h.alloc(0x100, 256, 1).unwrap();
    let x3 = h.alloc(0x200, 256, 1).unwrap();
    let _x4 = h.alloc(0x100, 256, 1).unwrap();
    h.free(x1.host).unwrap();
    h.free(x3.host).unwrap();
    // 0x180 rounds up to 0x200, which fits the hole of x3 exactly and the hole of x1 loosely.
    let y = h.alloc(0x180, 256, 1).unwrap();
    assert_eq!(y.apu, x3.apu);
    // 0x300 only fits the hole of x1.
    let z = h.alloc(0x300, 256, 1).unwrap();
    assert_eq!(z.apu, x1.apu);
}

#[test]
fn freed_blocks_merge_with_their_neighbours() {
    let mut h = small();
    let before = h.state(1).unwrap();
    let a = h.alloc(0x100, 256, 1).unwrap();
    let b = h.alloc(0x100, 256, 1).unwrap();
    let c = h.alloc(0x100, 256, 1).unwrap();
    h.free(b.host).unwrap();
    h.free(a.host).unwrap();
    h.free(c.host).unwrap();
    let after = h.state(1).unwrap();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.free_bytes, before.free_bytes);
    // One block spans from the end of the library blocks to the end of the heap.
    assert_eq!(after.largest_free_block, 0x20000 - 0x101C);
}

#[test]
fn free_accepts_only_the_start_of_a_live_allocation() {
    let mut h = small();
    let a = h.alloc(0x200, 256, 1).unwrap();
    assert_eq!(h.free(a.host.wrapping_add(1)), Err(E_INVALIDARG));
    assert_eq!(h.free(std::ptr::null()), Err(E_INVALIDARG));
    assert_eq!(h.free(&0u8), Err(E_INVALIDARG));
    assert_eq!(h.free(a.host), Ok(()));
    assert_eq!(h.free(a.host), Err(E_INVALIDARG));
}

#[test]
fn running_out_of_memory_is_reported() {
    let mut h = small();
    assert_eq!(h.alloc(0x20000, 256, 1).err(), Some(E_OUTOFMEMORY));
    assert_eq!(h.alloc(0xB000, 256, 2).err(), Some(E_OUTOFMEMORY));
    assert!(h.alloc(0xA000, 256, 2).is_ok());
    assert_eq!(h.alloc(256, 256, 2).err(), Some(E_OUTOFMEMORY));
}

#[test]
fn address_mapping_works_in_both_directions() {
    let mut h = small();
    let a = h.alloc(0x200, 256, 1).unwrap();
    assert_eq!(h.map_virtual(a.host), Some(a.apu));
    assert_eq!(h.map_virtual(a.host.wrapping_add(0x1FF)), Some(a.apu + 0x1FF));
    assert_eq!(h.map_virtual(a.host.wrapping_add(0x200)), None);
    assert_eq!(h.map_virtual(std::ptr::null()), None);
    assert_eq!(h.map_apu(a.apu), Some(a.host));
    assert_eq!(h.map_apu(a.apu + 0x10), Some(a.host.wrapping_add(0x10)));
    assert_eq!(h.map_apu(a.apu + 0x200), None);
    assert_eq!(h.map_apu(0), None);
    assert_eq!(h.map_apu_range(a.apu, 0x200), Some(a.host));
    assert_eq!(h.map_apu_range(a.apu, 0x201), None);
    assert!(h.is_valid(a.host, 0));
    assert!(h.is_valid(a.host, 256));
    assert!(!h.is_valid(a.host.wrapping_add(4), 256));
    assert!(h.is_valid(a.host.wrapping_add(4), 4));
    h.free(a.host).unwrap();
    assert_eq!(h.map_virtual(a.host), None);
    assert_eq!(h.map_apu(a.apu), None);
    assert!(!h.is_valid(a.host, 1));
}

#[test]
fn the_state_of_a_heap_that_is_not_1_or_2_is_invalid() {
    let h = small();
    assert_eq!(h.state(0).err(), Some(E_INVALIDARG));
    assert_eq!(h.state(3).err(), Some(E_INVALIDARG));
}
