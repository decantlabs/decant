//! Upload memory for list slot 148 (`AllocateUploadMemory(size, alignment)`).
//!
//! A console title takes a pointer from this method, writes constants or indices through it and
//! then hands the same number to the GPU as an address (a root CBV or SRV, an index buffer). On the
//! console the CPU and the GPU address are one. On the PC they are two numbers, so the proxy gives
//! out the CPU address of a persistently mapped upload buffer and translates it into the buffer's
//! GPU address wherever a list method takes an address (`translate`).
//!
//! The memory belongs to the command allocator the list records into and is released when that
//! allocator is reset, which is the time the title may reuse the allocator's memory too. A list
//! without a known allocator owns its memory itself.
//!
//! What this cannot do: an address stored inside memory the GPU reads (a constant buffer view in a
//! descriptor, a pointer inside another buffer) is not seen here and reaches the GPU as the CPU
//! address.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// The size of an owner's first chunk when the request is small. Every command allocator the title
/// records with holds at least one chunk until it is reset. A retail title keeps about 216 of them, most for
/// the 16 bytes of a user data mirror, where chunks of 4 MiB held 864 MiB and 1 MiB held 216 MiB.
pub const FIRST_CHUNK_BYTES: usize = 64 << 10;
/// The ceiling for automatic chunk growth when an owner needs more small blocks.
pub const CHUNK_BYTES: usize = 1 << 20;
/// The factor by which the next chunk grows from the previous chunk of the same owner.
pub const CHUNK_GROWTH: usize = 2;
/// The largest alignment honoured, the alignment of a buffer's start.
pub const MAX_ALIGN: usize = 64 << 10;
/// The smallest block and alignment handed out.
pub const MIN_BLOCK: usize = 16;

/// One upload buffer, persistently mapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// The PC resource, released when the owner is freed.
    pub res: usize,
    /// The CPU address of its first byte.
    pub cpu: usize,
    /// The GPU address of its first byte.
    pub gpu: u64,
    pub size: usize,
    pub used: usize,
}

/// The block size and alignment that are really used for a request.
pub fn normalize(size: usize, align: usize) -> (usize, usize) {
    let align = align.max(MIN_BLOCK).next_power_of_two().min(MAX_ALIGN);
    (size.max(MIN_BLOCK), align)
}

#[derive(Default)]
pub struct UploadMem {
    chunks: HashMap<usize, Vec<Chunk>>,
    /// CPU start to (size, GPU start, resource)
    ranges: BTreeMap<usize, (usize, u64, usize)>,
    owners: HashMap<usize, usize>,
}

impl UploadMem {
    /// The key that owns the memory of `list`: its allocator when known, else the list.
    pub fn owner_of(&self, list: usize) -> usize {
        self.owners.get(&list).copied().unwrap_or(list)
    }

    /// `list` records into `allocator` from now on (0 for none).
    pub fn set_allocator(&mut self, list: usize, allocator: usize) {
        if allocator == 0 {
            self.owners.remove(&list);
        } else {
            self.owners.insert(list, allocator);
        }
    }

    pub fn add_chunk(&mut self, owner: usize, chunk: Chunk) {
        self.ranges.insert(chunk.cpu, (chunk.size, chunk.gpu, chunk.res));
        self.chunks.entry(owner).or_default().push(chunk);
    }

    /// The size of the next buffer for `owner` and a request, a whole number of 64 KiB. Starts at
    /// `FIRST_CHUNK_BYTES`, doubles the last chunk up to `CHUNK_BYTES`, and always fits the request.
    pub fn chunk_bytes(&self, owner: usize, size: usize, align: usize) -> usize {
        let (size, align) = normalize(size, align);
        let target = match self.chunks.get(&owner).and_then(|c| c.last()) {
            Some(last) => last.size.saturating_mul(CHUNK_GROWTH).clamp(FIRST_CHUNK_BYTES, CHUNK_BYTES),
            None => FIRST_CHUNK_BYTES,
        };
        size.saturating_add(align).max(target).next_multiple_of(MAX_ALIGN)
    }

    /// A block of `size` bytes from the last chunk of `owner`: its CPU address.
    pub fn bump(&mut self, owner: usize, size: usize, align: usize) -> Option<usize> {
        let (size, align) = normalize(size, align);
        let chunk = self.chunks.get_mut(&owner)?.last_mut()?;
        let at = (chunk.cpu + chunk.used).next_multiple_of(align);
        let end = at.checked_add(size)?;
        if end > chunk.cpu + chunk.size {
            return None;
        }
        chunk.used = end - chunk.cpu;
        Some(at)
    }

    /// The GPU address of the CPU address `va`, when it lies in a chunk.
    pub fn to_gpu(&self, va: u64) -> Option<u64> {
        let (start, off) = self.find(va as usize, 0)?;
        let (_, gpu, _) = self.ranges[&start];
        Some(gpu + off as u64)
    }

    /// The resource and the offset inside it of `[va, va + len)`, when it lies in one chunk.
    pub fn locate(&self, va: u64, len: u64) -> Option<(usize, u64)> {
        let (start, off) = self.find(va as usize, usize::try_from(len).ok()?)?;
        Some((self.ranges[&start].2, off as u64))
    }

    fn find(&self, va: usize, len: usize) -> Option<(usize, usize)> {
        let (&start, &(size, _, _)) = self.ranges.range(..=va).next_back()?;
        let off = va - start;
        (off.checked_add(len)? <= size && off < size).then_some((start, off))
    }

    /// The bytes of all live chunks.
    pub fn bytes(&self) -> usize {
        self.ranges.values().map(|&(size, _, _)| size).sum()
    }

    /// Forgets the memory of `owner` and returns `(resource, allocated bytes)` to release.
    pub fn free_owner(&mut self, owner: usize) -> Vec<(usize, usize)> {
        let chunks = self.chunks.remove(&owner).unwrap_or_default();
        for c in &chunks {
            self.ranges.remove(&c.cpu);
        }
        chunks.into_iter().map(|c| (c.res, c.size)).collect()
    }
}

static MEM: Mutex<Option<UploadMem>> = Mutex::new(None);
/// Whether any chunk may exist, so that the common case takes no lock.
static ANY: AtomicBool = AtomicBool::new(false);

/// Runs `f` on the global table.
pub fn with<R>(f: impl FnOnce(&mut UploadMem) -> R) -> R {
    let mut g = MEM.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(UploadMem::default))
}

/// Marks that chunks exist (called after a chunk is added).
pub fn note_used() {
    ANY.store(true, Ordering::Release);
}

/// The GPU address for an address a title passes: the buffer's GPU address when `va` is upload memory
/// of slot 148, else `va` itself.
pub fn translate(va: u64) -> u64 {
    if !ANY.load(Ordering::Acquire) {
        return va;
    }
    with(|m| m.to_gpu(va)).unwrap_or(va)
}

/// The upload buffer and the byte offset that hold `[va, va + len)`, if it is upload memory.
pub fn locate(va: u64, len: u64) -> Option<(usize, u64)> {
    if !ANY.load(Ordering::Acquire) {
        return None;
    }
    with(|m| m.locate(va, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(res: usize, cpu: usize, gpu: u64, size: usize) -> Chunk {
        Chunk { res, cpu, gpu, size, used: 0 }
    }

    #[test]
    fn requests_are_normalized() {
        assert_eq!(normalize(0, 0), (16, 16));
        assert_eq!(normalize(100, 24), (100, 32));
        assert_eq!(normalize(8, 1 << 20), (16, MAX_ALIGN));
    }

    #[test]
    fn chunks_start_at_sixty_four_kib_and_grow_on_demand() {
        let mut m = UploadMem::default();
        let mut cpu = 0x100_0000usize;
        let mut gpu = 0x9000_0000u64;
        for (i, expected) in [
            FIRST_CHUNK_BYTES,
            128 << 10,
            256 << 10,
            512 << 10,
            CHUNK_BYTES,
            CHUNK_BYTES,
        ]
        .into_iter()
        .enumerate()
        {
            let size = m.chunk_bytes(1, 16, 16);
            assert_eq!(size, expected, "step {i}");
            m.add_chunk(1, chunk(i + 1, cpu, gpu, size));
            assert_eq!(m.bump(1, 16, 16), Some(cpu));
            cpu += size + MAX_ALIGN;
            gpu += (size + MAX_ALIGN) as u64;
        }
        assert_eq!(FIRST_CHUNK_BYTES, 64 << 10);

        // An owner with a 64 KiB chunk that asks for more than the doubled size gets a chunk that fits.
        let first = m.chunk_bytes(2, 16, 16);
        assert_eq!(first, FIRST_CHUNK_BYTES);
        m.add_chunk(2, chunk(0x20, 0x400_0000, 0xA000_0000, first));
        assert_eq!(m.bump(2, 16, 16), Some(0x400_0000));
        assert_eq!(m.bump(2, 200 << 10, 256), None);
        let fitted = m.chunk_bytes(2, 200 << 10, 256);
        assert_eq!(fitted, 256 << 10);
        m.add_chunk(2, chunk(0x21, 0x500_0004, 0xB000_0004, fitted));
        assert_eq!(m.bump(2, 200 << 10, 256), Some(0x500_0100));

        // Resetting an owner starts its next chunk back at 64 KiB.
        m.free_owner(1);
        assert_eq!(m.chunk_bytes(1, 16, 16), FIRST_CHUNK_BYTES);
    }

    #[test]
    fn large_uploads_keep_the_same_chunk_and_block_as_before() {
        let mut m = UploadMem::default();
        let small = m.chunk_bytes(2, 16, 16);
        m.add_chunk(2, chunk(0x10, 0x10_0000, 0x8000_0000, small));
        assert_eq!(m.bump(2, 16, 16), Some(0x10_0000));

        let mut base = 0x1000_0004usize;
        let mut gpu = 0xC000_0004u64;
        for (i, (req_size, req_align)) in [
            (CHUNK_BYTES, 16),
            (CHUNK_BYTES, 256),
            (CHUNK_BYTES, MAX_ALIGN),
            (CHUNK_BYTES + 1, 64),
            (3 * CHUNK_BYTES + 500, 512),
        ]
        .into_iter()
        .enumerate()
        {
            let (norm_size, norm_align) = normalize(req_size, req_align);
            let old_bytes = norm_size.saturating_add(norm_align).max(CHUNK_BYTES).next_multiple_of(MAX_ALIGN);
            assert_eq!(old_bytes % MAX_ALIGN, 0);
            assert!(old_bytes >= CHUNK_BYTES + norm_align);

            for owner in [10 + i, 2] {
                assert_eq!(m.bump(owner, req_size, req_align), None);
                let bytes = m.chunk_bytes(owner, req_size, req_align);
                assert_eq!(bytes, old_bytes);
                m.add_chunk(owner, chunk(0x100 + owner * 16 + i, base, gpu, bytes));
                let expected_at = base.next_multiple_of(norm_align);
                assert_eq!(m.bump(owner, req_size, req_align), Some(expected_at));
                assert_eq!(m.to_gpu(expected_at as u64), Some(gpu + (expected_at - base) as u64));
                base += bytes + MAX_ALIGN;
                gpu += (bytes + MAX_ALIGN) as u64;
            }
        }
    }

    #[test]
    fn blocks_are_aligned_and_run_out_with_the_chunk() {
        let mut m = UploadMem::default();
        assert_eq!(m.bump(1, 16, 16), None, "no chunk yet");
        m.add_chunk(1, chunk(0xA, 0x10_0000, 0x9000_0000, 0x1000));
        assert_eq!(m.bump(1, 20, 16), Some(0x10_0000));
        assert_eq!(m.bump(1, 16, 64), Some(0x10_0040), "aligned up past the first 20 bytes");
        assert_eq!(m.bump(1, 0x1000, 16), None, "does not fit any more");
        assert_eq!(m.bump(1, 16, 16), Some(0x10_0050), "a failed request takes nothing");
        assert_eq!(m.bump(2, 16, 16), None, "another owner has no chunk");
    }

    #[test]
    fn a_cpu_address_inside_a_chunk_becomes_the_matching_gpu_address() {
        let mut m = UploadMem::default();
        m.add_chunk(1, chunk(0xA, 0x10_0000, 0x9000_0000, 0x1000));
        m.add_chunk(2, chunk(0xB, 0x20_0000, 0xA000_0000, 0x2000));
        assert_eq!(m.to_gpu(0x10_0000), Some(0x9000_0000));
        assert_eq!(m.to_gpu(0x10_0FFF), Some(0x9000_0FFF));
        assert_eq!(m.to_gpu(0x10_1000), None, "one past the end");
        assert_eq!(m.to_gpu(0x0F_FFFF), None, "before the first chunk");
        assert_eq!(m.to_gpu(0x20_1800), Some(0xA000_1800));
        assert_eq!(m.to_gpu(0x9000_0000), None, "a GPU address is not translated again");
    }

    #[test]
    fn a_span_is_located_only_when_it_fits_one_chunk() {
        let mut m = UploadMem::default();
        m.add_chunk(1, chunk(0xA, 0x10_0000, 0x9000_0000, 0x1000));
        assert_eq!(m.locate(0x10_0100, 0x100), Some((0xA, 0x100)));
        assert_eq!(m.locate(0x10_0F00, 0x100), Some((0xA, 0xF00)));
        assert_eq!(m.locate(0x10_0F00, 0x101), None, "runs past the end");
        assert_eq!(m.locate(0x10_0000, u64::MAX), None, "no overflow");
    }

    #[test]
    fn freeing_an_owner_returns_its_resources_and_forgets_its_addresses() {
        let mut m = UploadMem::default();
        m.add_chunk(1, chunk(0xA, 0x10_0000, 0x9000_0000, 0x1000));
        m.add_chunk(1, chunk(0xC, 0x30_0000, 0xB000_0000, 0x1000));
        m.add_chunk(2, chunk(0xB, 0x20_0000, 0xA000_0000, 0x2000));
        let mut freed = m.free_owner(1);
        freed.sort_unstable();
        assert_eq!(freed, vec![(0xA, 0x1000), (0xC, 0x1000)]);
        assert_eq!(m.to_gpu(0x10_0000), None);
        assert_eq!(m.to_gpu(0x20_0000), Some(0xA000_0000), "the other owner keeps its memory");
        assert!(m.free_owner(1).is_empty());
    }

    #[test]
    fn a_list_owns_its_memory_until_an_allocator_is_known() {
        let mut m = UploadMem::default();
        assert_eq!(m.owner_of(0x77), 0x77);
        m.set_allocator(0x77, 0x99);
        assert_eq!(m.owner_of(0x77), 0x99);
        m.set_allocator(0x77, 0);
        assert_eq!(m.owner_of(0x77), 0x77);
    }
}
