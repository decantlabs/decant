//! The APU heaps (`docs/CONSOLE_LIBS.md` 2.3): host memory the audio processor can address with 32 bit APU addresses.
//!
//! `Heaps` holds the two public heaps and the small internal one. Each heap is one zeroed host allocation aligned to 64 KiB
//! whose size is a multiple of 4 KiB, with a best fit free list. All of it is pure Rust on `Heaps`, so tests make as many
//! independent sets as they need. The exports use one process wide `Heaps` behind a lock (`global`).
#![allow(unsafe_code)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use crate::abi::*;

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

#[derive(Clone, Copy)]
struct Block {
    /// Bytes the block occupies (the size rounded up to the alignment).
    len: u32,
    /// Bytes the caller asked for.
    requested: u32,
}

/// One heap: a host allocation, the free list and the live allocations, both keyed by offset.
pub struct Heap {
    apu_base: u32,
    size: u32,
    mem: *mut u8,
    layout: Layout,
    free: BTreeMap<u32, u32>,
    live: BTreeMap<u32, Block>,
}

// SAFETY: the memory is owned by the heap and only reached through `&self` or `&mut self`.
unsafe impl Send for Heap {}

impl Heap {
    fn new(apu_base: u32, size: u32) -> Option<Heap> {
        let layout = Layout::from_size_align(size as usize, HEAP_HOST_ALIGN).ok()?;
        // SAFETY: the layout has a nonzero size, `size` is at least 0x1000.
        let mem = unsafe { alloc_zeroed(layout) };
        if mem.is_null() {
            return None;
        }
        Some(Heap { apu_base, size, mem, layout, free: BTreeMap::from([(0, size)]), live: BTreeMap::new() })
    }

    pub fn apu_base(&self) -> u32 {
        self.apu_base
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// Best fit allocation of `size` bytes aligned to `alignment` (a power of two), as an offset in the heap.
    fn alloc(&mut self, size: u32, alignment: u32) -> Option<u32> {
        let len = u32::try_from(align_up(u64::from(size), u64::from(alignment))).ok()?;
        let mut best: Option<(u32, u32, u32)> = None; // (free length, free offset, start offset)
        for (&off, &flen) in &self.free {
            let apu = u64::from(self.apu_base) + u64::from(off);
            let start = align_up(apu, u64::from(alignment)) - u64::from(self.apu_base);
            let gap = start - u64::from(off);
            if gap + u64::from(len) <= u64::from(flen) && best.is_none_or(|(l, o, _)| (flen, off) < (l, o)) {
                best = Some((flen, off, start as u32));
            }
        }
        let (flen, off, start) = best?;
        self.free.remove(&off);
        if start > off {
            self.free.insert(off, start - off);
        }
        let end = start + len;
        if off + flen > end {
            self.free.insert(end, off + flen - end);
        }
        self.live.insert(start, Block { len, requested: size });
        // SAFETY: [start, start + len) is inside the heap allocation and owned by no other block.
        unsafe { std::ptr::write_bytes(self.mem.add(start as usize), 0, len as usize) };
        Some(start)
    }

    fn free_block(&mut self, offset: u32) -> bool {
        let Some(block) = self.live.remove(&offset) else { return false };
        let (mut start, mut len) = (offset, block.len);
        if let Some((&poff, &plen)) = self.free.range(..start).next_back()
            && poff + plen == start
        {
            self.free.remove(&poff);
            start = poff;
            len += plen;
        }
        if let Some(&nlen) = self.free.get(&(start + len)) {
            self.free.remove(&(start + len));
            len += nlen;
        }
        self.free.insert(start, len);
        true
    }

    /// The live block containing `offset`, as (start, block).
    fn containing(&self, offset: u32) -> Option<(u32, Block)> {
        let (&start, &block) = self.live.range(..=offset).next_back()?;
        (offset < start + block.len).then_some((start, block))
    }

    pub fn state(&self) -> ApuHeapState {
        let free: u32 = self.free.values().sum();
        let used: u32 = self.live.values().map(|b| b.requested).sum();
        let overhead: u32 = self.live.values().map(|b| b.len - b.requested).sum();
        ApuHeapState {
            free_bytes: free,
            used_bytes: used,
            overhead_bytes: overhead,
            largest_free_block: self.free.values().copied().max().unwrap_or(0),
            allocations: self.live.len() as u32,
        }
    }

    fn host(&self, offset: u32) -> *mut u8 {
        // SAFETY: callers pass an offset inside the heap.
        unsafe { self.mem.add(offset as usize) }
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        // SAFETY: `mem` came from `alloc_zeroed` with this layout.
        unsafe { dealloc(self.mem, self.layout) };
    }
}

/// The two public heaps (numbers 1 and 2) and the internal one.
pub struct Heaps {
    heaps: [Heap; 3],
}

/// A successful allocation: the host pointer and the APU address.
#[derive(Clone, Copy, Debug)]
pub struct Allocation {
    pub host: *mut u8,
    pub apu: u32,
}

impl Heaps {
    /// Creates the heaps as `ApuCreateHeap(size1, size2)` does, including the blocks the library takes for itself.
    pub fn new(size1: u32, size2: u32) -> Result<Heaps, Hresult> {
        let size2 = if size2 == 0 { DEFAULT_HEAP2_SIZE } else { size2 };
        let sizes = [size1, size2.checked_add(HEAP2_EXTRA).ok_or(E_OUTOFMEMORY)?, INTERNAL_HEAP_SIZE];
        let mut base = u64::from(HEAP1_APU_BASE);
        let mut heaps = Vec::with_capacity(3);
        for size in sizes {
            let size = align_up(u64::from(size), 0x1000);
            if base + size > u64::from(u32::MAX) {
                return Err(E_OUTOFMEMORY);
            }
            heaps.push(Heap::new(base as u32, size as u32).ok_or(E_OUTOFMEMORY)?);
            base = align_up(base + size, HEAP_HOST_ALIGN as u64);
        }
        let Ok(heaps) = <[Heap; 3]>::try_from(heaps) else { return Err(E_FAIL) };
        let mut set = Heaps { heaps };
        // The blocks of the library itself.
        let own = [(0usize, 0x1000, 0x100), (0, 0x14, 4), (0, 8, 4), (2, 0x4C, 4)];
        for (heap, size, alignment) in own {
            set.heaps[heap].alloc(size, alignment).ok_or(E_OUTOFMEMORY)?;
        }
        Ok(set)
    }

    /// Heap 1 or 2 as an index into the array, `None` for any other number.
    fn index(heap: u32) -> Option<usize> {
        (heap == 1 || heap == 2).then(|| heap as usize - 1)
    }

    /// `ApuAlloc` on a created set.
    pub fn alloc(&mut self, size: u32, alignment: u32, heap: u32) -> Result<Allocation, Hresult> {
        let index = Self::index(heap).ok_or(E_INVALIDARG)?;
        if size == 0 || alignment == 0 || !alignment.is_power_of_two() || !alignment.is_multiple_of(4) {
            return Err(E_INVALIDARG);
        }
        let alignment = alignment.max(MIN_ALIGNMENT);
        let h = &mut self.heaps[index];
        let offset = h.alloc(size, alignment).ok_or(E_OUTOFMEMORY)?;
        Ok(Allocation { host: h.host(offset), apu: h.apu_base + offset })
    }

    /// `ApuFree`: only the start of a live allocation is accepted.
    pub fn free(&mut self, host: *const u8) -> Result<(), Hresult> {
        for h in &mut self.heaps {
            let start = h.mem as usize;
            let addr = host as usize;
            if addr >= start && addr < start + h.size as usize && h.free_block((addr - start) as u32) {
                return Ok(());
            }
        }
        Err(E_INVALIDARG)
    }

    /// The APU address of any byte inside a live allocation.
    pub fn map_virtual(&self, host: *const u8) -> Option<u32> {
        for h in &self.heaps {
            let start = h.mem as usize;
            let addr = host as usize;
            if addr >= start && addr < start + h.size as usize {
                let offset = (addr - start) as u32;
                return h.containing(offset).map(|_| h.apu_base + offset);
            }
        }
        None
    }

    /// The host address of an APU address inside a live allocation.
    pub fn map_apu(&self, apu: u32) -> Option<*mut u8> {
        self.map_apu_range(apu, 1)
    }

    /// The host address of `len` bytes at `apu`, when they lie inside one live allocation.
    pub fn map_apu_range(&self, apu: u32, len: u32) -> Option<*mut u8> {
        for h in &self.heaps {
            if apu >= h.apu_base && apu - h.apu_base < h.size {
                let offset = apu - h.apu_base;
                let (start, block) = h.containing(offset)?;
                return (u64::from(offset) + u64::from(len) <= u64::from(start) + u64::from(block.len)).then(|| h.host(offset));
            }
        }
        None
    }

    pub fn is_valid(&self, host: *const u8, alignment: u32) -> bool {
        let alignment = alignment.max(1);
        self.map_virtual(host).is_some_and(|apu| apu % alignment == 0)
    }

    /// `ApuHeapGetState` for heap 1 or 2.
    pub fn state(&self, heap: u32) -> Result<ApuHeapState, Hresult> {
        Ok(self.heaps[Self::index(heap).ok_or(E_INVALIDARG)?].state())
    }

    /// The heap descriptor (APU base and size) of heap 1, 2 or 3 for the internal one.
    pub fn extent(&self, index: usize) -> (u32, u32) {
        (self.heaps[index].apu_base, self.heaps[index].size)
    }
}

static GLOBAL: Mutex<Option<Heaps>> = Mutex::new(None);

/// The process wide lock of the heap functions.
pub fn global() -> MutexGuard<'static, Option<Heaps>> {
    GLOBAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The heaps, created with the default sizes when they do not exist yet.
pub fn with_heaps<R>(f: impl FnOnce(&mut Heaps) -> R) -> Result<R, Hresult> {
    let mut guard = global();
    if guard.is_none() {
        *guard = Some(Heaps::new(DEFAULT_HEAP1_SIZE, 0)?);
    }
    Ok(f(guard.as_mut().ok_or(E_FAIL)?))
}


#[cfg(test)]
mod tests;
