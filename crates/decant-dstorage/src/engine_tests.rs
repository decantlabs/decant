//! Console requests with a physical page destination, run by a real queue of the CPU backend on
//! synthetic files, into a synthetic page pool. The pool of `xmem.dll` itself (writes into pages that
//! are mapped only later) is tested in `decant-xmem`.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::AtomicU64;
use std::sync::Mutex;
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::*;
use crate::pages::PAGE_BYTES as P;

/// A pool of 64 KiB pages by frame number. A write to a frame it does not hold fails whole, as the pool
/// of `xmem.dll` does. Each write notes the thread it ran on.
struct TestPool {
    pages: Mutex<HashMap<u64, Vec<u8>>>,
    writers: Mutex<Vec<ThreadId>>,
}

impl TestPool {
    fn new(frames: &[u64]) -> Arc<TestPool> {
        Arc::new(TestPool { pages: Mutex::new(frames.iter().map(|&f| (f, vec![0u8; P])).collect()), writers: Mutex::new(Vec::new()) })
    }

    fn page(&self, frame: u64) -> Vec<u8> {
        lock(&self.pages)[&frame].clone()
    }
}

impl PagePool for TestPool {
    fn write(&self, frames: &[u64], offset: usize, data: &[u8]) -> Result<Vec<(usize, usize)>, HRESULT> {
        lock(&self.writers).push(std::thread::current().id());
        let mut pages = lock(&self.pages);
        let count = (offset + data.len()).div_ceil(P);
        if offset >= P || frames.len() < count || frames[..count].iter().any(|f| !pages.contains_key(f)) {
            return Err(E_INVALIDARG);
        }
        let mut at = 0;
        for (k, frame) in frames[..count].iter().enumerate() {
            let start = if k == 0 { offset } else { 0 };
            let n = (P - start).min(data.len() - at);
            pages.get_mut(frame).expect("checked above")[start..start + n].copy_from_slice(&data[at..at + n]);
            at += n;
        }
        Ok(Vec::new())
    }
}

/// A file of the temporary folder, removed on drop.
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(name: &str, contents: &[u8]) -> TempFile {
        let path = std::env::temp_dir().join(format!("dstorage_pages_{}_{name}.bin", std::process::id()));
        std::fs::write(&path, contents).expect("the temporary file is written");
        TempFile(path)
    }

    fn wide(&self) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        self.0.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A stand in for an `ID3D12Fence`: `Signal` (slot 10) stores the value and the thread it ran on.
#[repr(C)]
struct TestFence {
    vtbl: *const usize,
    refs: AtomicU32,
    value: AtomicU64,
    signaller: Mutex<Option<ThreadId>>,
}

unsafe extern "system" fn fence_add_ref(this: *mut TestFence) -> u32 {
    // SAFETY: called through the vtable of a live `TestFence`.
    unsafe { (*this).refs.fetch_add(1, Ordering::SeqCst) + 1 }
}

unsafe extern "system" fn fence_release(this: *mut TestFence) -> u32 {
    // SAFETY: as above. The test owns the object, so it is never freed here.
    unsafe { (*this).refs.fetch_sub(1, Ordering::SeqCst) - 1 }
}

unsafe extern "system" fn fence_signal(this: *mut TestFence, value: u64) -> HRESULT {
    // SAFETY: as above.
    unsafe {
        *lock(&(*this).signaller) = Some(std::thread::current().id());
        (*this).value.store(value, Ordering::SeqCst);
    }
    S_OK
}

unsafe extern "system" fn fence_other() -> HRESULT {
    E_NOTIMPL
}

impl TestFence {
    fn new() -> Box<TestFence> {
        static VTBL: OnceLock<[usize; 11]> = OnceLock::new();
        let vtbl = VTBL.get_or_init(|| {
            let mut v = [fence_other as *const () as usize; 11];
            v[1] = fence_add_ref as *const () as usize;
            v[2] = fence_release as *const () as usize;
            v[10] = fence_signal as *const () as usize;
            v
        });
        Box::new(TestFence { vtbl: vtbl.as_ptr(), refs: AtomicU32::new(1), value: AtomicU64::new(0), signaller: Mutex::new(None) })
    }
}

/// A factory of the CPU backend over `pool`, a queue and a status array of four slots.
fn queue_over(pool: Arc<TestPool>) -> (Ref<Factory>, Ref<Queue>, Ref<Status>) {
    let factory = Ref::new(Factory::new(Backend::cpu(), pool));
    let queue = factory.create_queue(16).expect("a queue");
    let status = factory.create_status_array(4).expect("a status array");
    (factory, queue, status)
}

/// A console request of `file` to the page array `frames`.
fn request(file: &Ref<File>, options: u64, frames: &[u64], page_offset: u64, destination_size: usize, offset: u64, source_size: usize) -> DecantConsoleRead {
    let mut r = DecantConsoleRead::zeroed();
    r.options = options;
    r.destination = frames.as_ptr().cast_mut().cast();
    r.destination_page_offset = page_offset;
    r.destination_size = destination_size as u32;
    r.file = file.as_ptr().cast();
    r.offset = offset;
    r.source_size = source_size as u32;
    r
}

/// Puts a status entry in slot `index`, submits, and waits for the slot.
fn finish(queue: &Queue, status: &Ref<Status>, index: u32) -> HRESULT {
    // SAFETY: the status array is one of ours.
    assert_eq!(unsafe { queue.enqueue_status(status.as_ptr().cast(), index) }, S_OK);
    assert_eq!(queue.submit(), S_OK);
    let start = Instant::now();
    while status.result(index) == E_PENDING {
        assert!(start.elapsed() < Duration::from_secs(10), "the status slot completes");
        std::thread::sleep(Duration::from_millis(1));
    }
    status.result(index)
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + i / 251) as u8).collect()
}

fn zlib_of(data: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(data).expect("compresses");
    encoder.finish().expect("finishes")
}

const PAGES: u64 = console_options::DESTINATION_PAGES;

#[test]
fn a_zlib_request_fills_scattered_pages_from_the_page_offset_and_completes_on_the_worker() {
    let plain = pattern(2 * P + 0x3000);
    let packed = zlib_of(&plain);
    // The chunk at a 256 byte offset of the file, as a retail title's chunk tables place them.
    let mut contents = vec![0xEEu8; 0x100];
    contents.extend_from_slice(&packed);
    let file_on_disk = TempFile::new("zlib", &contents);
    let frames = [0x50u64, 0x10, 0x90];
    let pool = TestPool::new(&[0x10, 0x50, 0x90, 0xA0]);
    let (factory, queue, status) = queue_over(pool.clone());
    let file = factory.open_file(file_on_disk.wide().as_ptr()).expect("the file opens");
    let start = 0x1234;
    let r = request(&file, PAGES | console_options::ZLIB, &frames, start as u64, plain.len(), 0x100, packed.len());
    assert_eq!(queue.enqueue_console(&r), S_OK);
    // SAFETY: plain Win32 calls on an event of ours.
    let event = Handle(unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) });
    assert_eq!(queue.enqueue_event(event.0), S_OK);
    let fence = TestFence::new();
    let fence_ptr = std::ptr::from_ref(&*fence).cast_mut().cast::<c_void>();
    // SAFETY: a stand in with the fence's vtable slots.
    assert_eq!(unsafe { queue.enqueue_fence(fence_ptr, 9) }, S_OK);
    assert_eq!(finish(&queue, &status, 0), S_OK);
    // SAFETY: as above.
    assert_eq!(unsafe { WaitForSingleObject(event.0, 10_000) }, WAIT_OBJECT_0, "the event is set");
    let start_wait = Instant::now();
    while fence.value.load(Ordering::SeqCst) != 9 {
        assert!(start_wait.elapsed() < Duration::from_secs(10), "the fence is signalled");
        std::thread::sleep(Duration::from_millis(1));
    }
    let (first, second, third) = (pool.page(0x50), pool.page(0x10), pool.page(0x90));
    assert!(first[..start].iter().all(|&b| b == 0), "nothing before the page offset");
    assert_eq!(&first[start..], &plain[..P - start]);
    assert_eq!(&second[..], &plain[P - start..2 * P - start]);
    let rest = plain.len() - (2 * P - start);
    assert_eq!(&third[..rest], &plain[2 * P - start..], "the request ends in the middle of its last page");
    assert!(third[rest..].iter().all(|&b| b == 0));
    assert!(pool.page(0xA0).iter().all(|&b| b == 0), "a page the array does not name");
    // The pages were written (checked above) before the fence was signalled: the queue's worker
    // signals only after the decode pool has finished the reads before the fence.
    let writer = lock(&pool.writers)[0];
    assert_ne!(writer, std::thread::current().id(), "the write runs on a decode thread");
    let signaller = lock(&fence.signaller).expect("the fence is signalled");
    assert_ne!(signaller, std::thread::current().id(), "the fence is signalled by the queue's worker");
    assert_eq!(fence.refs.load(Ordering::SeqCst), 1, "the queue gave its fence reference back");
}

#[test]
fn an_uncompressed_request_may_end_in_the_middle_of_its_only_page() {
    let plain = pattern(0x9000);
    let file_on_disk = TempFile::new("copy", &plain);
    let pool = TestPool::new(&[0x20]);
    let (factory, queue, status) = queue_over(pool.clone());
    let file = factory.open_file(file_on_disk.wide().as_ptr()).expect("the file opens");
    // One entry: a destination of 0x4000 bytes from 0x2000 needs no second frame.
    let frames = [0x20u64];
    assert_eq!(queue.enqueue_console(&request(&file, PAGES, &frames, 0x2000, 0x4000, 0x1000, 0x4000)), S_OK);
    assert_eq!(finish(&queue, &status, 0), S_OK);
    let page = pool.page(0x20);
    assert_eq!(&page[0x2000..0x6000], &plain[0x1000..0x5000]);
    assert!(page[..0x2000].iter().chain(&page[0x6000..]).all(|&b| b == 0));
}

#[test]
fn a_frame_the_pool_does_not_hold_fails_the_status_and_writes_nothing() {
    let plain = pattern(P + 0x100);
    let file_on_disk = TempFile::new("missing", &plain);
    let pool = TestPool::new(&[0x30]);
    let (factory, queue, status) = queue_over(pool.clone());
    let file = factory.open_file(file_on_disk.wide().as_ptr()).expect("the file opens");
    let frames = [0x30u64, 0x40];
    assert_eq!(queue.enqueue_console(&request(&file, PAGES, &frames, 0, plain.len(), 0, plain.len())), S_OK);
    assert_eq!(finish(&queue, &status, 0), E_INVALIDARG);
    assert!(pool.page(0x30).iter().all(|&b| b == 0));
    // The next status entry starts clean.
    assert_eq!(queue.enqueue_console(&request(&file, PAGES, &frames[..1], 0, 0x100, 0, 0x100)), S_OK);
    assert_eq!(finish(&queue, &status, 1), S_OK);
    assert_eq!(&pool.page(0x30)[..0x100], &plain[..0x100]);
}

#[test]
fn zlib_data_that_does_not_fill_the_destination_fails_and_writes_nothing() {
    let plain = pattern(0x8000);
    let packed = zlib_of(&plain);
    let file_on_disk = TempFile::new("short", &packed);
    let pool = TestPool::new(&[0x10]);
    let (factory, queue, status) = queue_over(pool.clone());
    let file = factory.open_file(file_on_disk.wide().as_ptr()).expect("the file opens");
    let frames = [0x10u64];
    assert_eq!(queue.enqueue_console(&request(&file, PAGES | console_options::ZLIB, &frames, 0, P, 0, packed.len())), S_OK);
    assert!(finish(&queue, &status, 0) < 0);
    assert!(pool.page(0x10).iter().all(|&b| b == 0));
    assert!(lock(&pool.writers).is_empty(), "the pool is not called");
}

#[test]
fn a_bc1_request_fills_pages() {
    let want = pattern(0x6000); // BC1 blocks of 8 bytes, over the end of the first page
    let chunk = decant_bcpack::encode::pack_plain(decant_bcpack::Format::Bc1, &want, 300);
    let file_on_disk = TempFile::new("bc1", &chunk);
    let pool = TestPool::new(&[0x70, 0x60]);
    let (factory, queue, status) = queue_over(pool.clone());
    let file = factory.open_file(file_on_disk.wide().as_ptr()).expect("the file opens");
    let frames = [0x70u64, 0x60];
    let start = P - 0x1000;
    let mode = 1u64 << console_options::BCPACK_SHIFT;
    assert_eq!(queue.enqueue_console(&request(&file, PAGES | mode, &frames, start as u64, want.len(), 0, chunk.len())), S_OK);
    assert_eq!(finish(&queue, &status, 0), S_OK);
    assert_eq!(&pool.page(0x70)[start..], &want[..P - start]);
    assert_eq!(&pool.page(0x60)[..want.len() - (P - start)], &want[P - start..]);
}

/// One request of every BCPack mode to an address, in one submission, from one file.
#[test]
fn bcpack_requests_of_every_mode_fill_their_destinations() {
    let mut chunks = Vec::new();
    for format in decant_bcpack::Format::ALL {
        let layout = format.layout();
        let blocks = 700;
        // Varied symbols that fit every lane; lane 15 stays 0 (a BC7 control code and BC6H mode 1 that exist).
        let records: Vec<u8> = (0..blocks * layout.stride())
            .map(|k| {
                let lane = (k % layout.stride()) / layout.per;
                if lane == 15 { 0 } else { ((k * 7 + k / 13) % (1 << (layout.groups[lane] + 1))) as u8 }
            })
            .collect();
        let stream = decant_bcpack::encode::stream(layout, &records, 250);
        let (want, _) = decant_bcpack::decode_to_vec(format, &stream, blocks * format.block_bytes()).expect("the stream decodes");
        chunks.push((format, stream, want));
    }
    // Console requests with a codec start on 16 byte offsets.
    let padded = |n: usize| n.next_multiple_of(16);
    let contents: Vec<u8> = chunks.iter().flat_map(|(_, stream, _)| [stream.clone(), vec![0; padded(stream.len()) - stream.len()]].concat()).collect();
    let file_on_disk = TempFile::new("bcp", &contents);
    let pool = TestPool::new(&[]);
    let (factory, queue, status) = queue_over(pool);
    let file = factory.open_file(file_on_disk.wide().as_ptr()).expect("the file opens");
    let mut outputs: Vec<Vec<u8>> = chunks.iter().map(|(_, _, want)| vec![0xCD; want.len()]).collect();
    let mut offset = 0;
    for ((format, stream, _), out) in chunks.iter().zip(outputs.iter_mut()) {
        let mut r = request(&file, u64::from(format.mode()) << console_options::BCPACK_SHIFT, &[], 0, out.len(), offset as u64, stream.len());
        r.destination = out.as_mut_ptr().cast();
        assert_eq!(queue.enqueue_console(&r), S_OK);
        offset += padded(stream.len());
    }
    assert_eq!(finish(&queue, &status, 0), S_OK);
    for ((format, _, want), got) in chunks.iter().zip(&outputs) {
        assert!(got == want, "{format:?}");
    }
}

#[test]
fn options_that_are_not_implemented_are_refused_by_name() {
    use console_options::*;
    assert_eq!(console_refusal(PAGES | ZLIB), None);
    assert_eq!(console_refusal(PAGES | (7 << BCPACK_SHIFT)), None, "BC7, the mode a retail title asked for");
    assert_eq!(console_refusal(0), None);
    assert_eq!(console_refusal(PAGES | ZLIB | (3 << SWIZZLE_SHIFT)).as_deref(), Some("swizzle mode 3 (bits 5 to 20)"));
    assert_eq!(console_refusal(MEMORY_SOURCE).as_deref(), Some("memory source (bit 22)"));
    assert_eq!(console_refusal(MEMORY_SOURCE | SOURCE_PAGES).as_deref(), Some("source in physical pages (bit 23)"));
    assert_eq!(console_refusal(1 << 30).as_deref(), Some("undeclared option bits 0x40000000"));
    assert_eq!(console_refusal(8 << BCPACK_SHIFT).as_deref(), Some("BCPack mode 8"));
    assert_eq!(console_refusal(ZLIB | (1 << BCPACK_SHIFT)).as_deref(), Some("zlib then BCPack mode 1"));
    let pool = TestPool::new(&[0x10]);
    let (_factory, queue, _status) = queue_over(pool);
    let frames = [0x10u64];
    let mut r = DecantConsoleRead::zeroed();
    r.options = PAGES | ZLIB | (1 << SWIZZLE_SHIFT);
    r.destination = frames.as_ptr().cast_mut().cast();
    r.destination_size = 16;
    r.source_size = 16;
    assert_eq!(queue.enqueue_console(&r), E_NOTIMPL, "refused at enqueue, before the file is looked at");
    r.options = PAGES;
    r.destination = std::ptr::null_mut();
    assert_eq!(queue.enqueue_console(&r), E_INVALIDARG, "a page destination needs its array");
}

#[test]
fn batched_reads_preserve_barrier_ordering_and_propagate_eof_errors() {
    let chunk_size = 4096usize;
    let count = 72usize;
    let plain = pattern(chunk_size * count);
    let file_on_disk = TempFile::new("batched", &plain);
    let pool = TestPool::new(&[]);
    let factory = Ref::new(Factory::new(Backend::cpu(), pool));
    let queue = factory.create_queue(128).expect("a queue");
    let status = factory.create_status_array(4).expect("a status array");
    let file = factory.open_file(file_on_disk.wide().as_ptr()).expect("the file opens");

    let mut outputs: Vec<Vec<u8>> = (0..count).map(|_| vec![0xAA; chunk_size]).collect();
    for (i, out) in outputs.iter_mut().enumerate() {
        let mut r = request(&file, 0, &[], 0, chunk_size, (i * chunk_size) as u64, chunk_size);
        r.destination = out.as_mut_ptr().cast();
        assert_eq!(queue.enqueue_console(&r), S_OK);
    }
    assert_eq!(finish(&queue, &status, 0), S_OK);
    for (i, out) in outputs.iter().enumerate() {
        assert_eq!(out, &plain[i * chunk_size..(i + 1) * chunk_size]);
    }

    // Mix a valid read with an EOF read in one batch; status slot 1 must observe the error,
    // and status slot 2 after a clean read must observe S_OK.
    let mut ok_buf = vec![0u8; chunk_size];
    let mut eof_buf = vec![0u8; chunk_size];
    let mut r_ok = request(&file, 0, &[], 0, chunk_size, 0, chunk_size);
    r_ok.destination = ok_buf.as_mut_ptr().cast();
    let mut r_eof = request(&file, 0, &[], 0, chunk_size, plain.len() as u64, chunk_size);
    r_eof.destination = eof_buf.as_mut_ptr().cast();
    assert_eq!(queue.enqueue_console(&r_ok), S_OK);
    assert_eq!(queue.enqueue_console(&r_eof), S_OK);
    assert!(finish(&queue, &status, 1) < 0);
    assert_eq!(&ok_buf, &plain[..chunk_size]);

    let mut clean_buf = vec![0u8; chunk_size];
    let mut r_clean = request(&file, 0, &[], 0, chunk_size, chunk_size as u64, chunk_size);
    r_clean.destination = clean_buf.as_mut_ptr().cast();
    assert_eq!(queue.enqueue_console(&r_clean), S_OK);
    assert_eq!(finish(&queue, &status, 2), S_OK);
    assert_eq!(&clean_buf, &plain[chunk_size..2 * chunk_size]);
}

#[test]
fn write_hooks_stay_no_op_when_xmem_is_not_loaded() {
    let mut scratch = [0u8; 64];
    let pinned = begin_written(scratch.as_mut_ptr().cast(), scratch.len() as u32);
    assert!(!pinned);
    finish_written(scratch.as_mut_ptr().cast(), scratch.len() as u32);
    note_written(scratch.as_mut_ptr().cast(), scratch.len() as u32);
}

#[test]
fn frame_zero_unmapped_pages_skip_cleanly_without_failing_status() {
    let pool = TestPool::new(&[0x30, 0x40]);
    let (factory, queue, status) = queue_over(pool.clone());
    let plain = pattern(3 * P + 0x2000);
    let on_disk = TempFile::new("frame_zero_holes", &plain);
    let file = factory.open_file(on_disk.wide().as_ptr()).expect("the file opens");

    // 1) All-zero frames (`NO_FRAME`) skip cleanly and return `S_OK` without touching pool pages.
    let zero_frames = [0u64, 0u64];
    assert_eq!(queue.enqueue_console(&request(&file, PAGES, &zero_frames, 0, P + 0x2000, 0, P + 0x2000)), S_OK);
    assert_eq!(finish(&queue, &status, 0), S_OK);
    assert!(pool.page(0x30).iter().all(|&b| b == 0));
    assert!(pool.page(0x40).iter().all(|&b| b == 0));

    // 2) Mixed `[0, 0x30, 0, 0x40]` with non-zero offset writes ONLY the mapped runs accurately.
    let offset = 0x1000usize;
    let total = 3 * P;
    let mixed = [0u64, 0x30, 0, 0x40];
    assert_eq!(queue.enqueue_console(&request(&file, PAGES, &mixed, offset as u64, total, 0, total)), S_OK);
    assert_eq!(finish(&queue, &status, 1), S_OK);
    let p0_end = P - offset; // bytes skipped in frame[0]=0
    assert_eq!(&pool.page(0x30)[..], &plain[p0_end..p0_end + P]);
    let p2_end = p0_end + 2 * P; // bytes skipped across frame[2]=0
    let tail = total - p2_end;
    assert_eq!(&pool.page(0x40)[..tail], &plain[p2_end..total]);
    assert!(pool.page(0x40)[tail..].iter().all(|&b| b == 0));

    // 3) Mixed `[0, 0x99]` where a non-zero frame is missing in the pool STILL fails with `E_INVALIDARG`.
    let bad = [0u64, 0x99];
    assert_eq!(queue.enqueue_console(&request(&file, PAGES, &bad, 0, P + 0x100, 0, P + 0x100)), S_OK);
    assert_eq!(finish(&queue, &status, 2), E_INVALIDARG);
}

#[test]
fn frames_populated_before_submit_survive_array_overwrite_after_submit() {
    let pool = TestPool::new(&[0x10, 0x20]);
    let (factory, queue, status) = queue_over(pool.clone());
    let plain = pattern(2 * P);
    let on_disk = TempFile::new("submit_snapshot", &plain);
    let file = factory.open_file(on_disk.wide().as_ptr()).expect("the file opens");

    // Enqueue with `[0, 0]`, populate before `submit()`, then wipe & free the backing array
    // immediately after `submit()` while worker I/O runs asynchronously.
    let mut frames = vec![0u64, 0u64];
    assert_eq!(queue.enqueue_console(&request(&file, PAGES, &frames, 0, plain.len(), 0, plain.len())), S_OK);
    frames[0] = 0x10;
    frames[1] = 0x20;
    // SAFETY: `status` is a live `Status` COM object.
    unsafe { assert_eq!(queue.enqueue_status(status.as_ptr().cast(), 0), S_OK) };
    assert_eq!(queue.submit(), S_OK);
    frames.fill(0);
    drop(frames);

    // SAFETY: `status` is a live `Status` COM object.
    unsafe {
        let raw_status = status.as_ptr().cast();
        while (vtable::<StatusVtbl>(raw_status).result)(raw_status, 0) == E_PENDING {
            std::thread::yield_now();
        }
        assert_eq!((vtable::<StatusVtbl>(raw_status).result)(raw_status, 0), S_OK);
    }
    assert_eq!(pool.page(0x10), &plain[..P]);
    assert_eq!(pool.page(0x20), &plain[P..]);
}
