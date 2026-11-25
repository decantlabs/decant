//! `dstoragetest <dstorage_xs.dll>`: the checks of the former `tests/dstoragetest.cpp`, with the
//! same lines in the same order and the same exit codes (0 all passed, 1 a check failed, 2 bad
//! arguments or a DLL that does not load). The C++ test linked the DLL's import library, here the
//! DLL is loaded by path.
//!
//! The test makes its compressed data itself with libdeflate and its GDeflate compressor, which
//! `build.rs` links only when `DECANT_GDEFLATE_ROOT` is set. Without it the program prints `SKIP`
//! and passes.

#![allow(unsafe_code)]

#[cfg(all(windows, decant_gdeflate))]
#[path = "../testapi.rs"]
mod testapi;

#[cfg(not(all(windows, decant_gdeflate)))]
fn main() {
    println!("SKIP: dstoragetest needs GDeflate to make its test data (build with DECANT_GDEFLATE_ROOT set)");
}

#[cfg(all(windows, decant_gdeflate))]
fn main() -> std::process::ExitCode {
    let args = decant_testkit::args();
    if args.len() != 1 {
        return std::process::ExitCode::from(2);
    }
    let Some(dll) = testapi::Dll::load(&args[0]) else {
        return std::process::ExitCode::from(2);
    };
    std::process::ExitCode::from(run::run(&dll))
}

#[cfg(all(windows, decant_gdeflate))]
mod run {
    use std::ffi::c_void;
    use std::time::Duration;

    use dstorage_xs::libdeflate::{gdeflate_compress, zlib_compress};
    use dstorage_xs::decant::*;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
    use windows_sys::Win32::Storage::FileSystem::WriteFile;
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::Pipes::{ConnectNamedPipe, CreateNamedPipeW};
    use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    use crate::check;
    use crate::testapi::{Dll, Obj, TempFile, TestFence, wait_for, wait_status_default};

    const TIMEOUT: Duration = Duration::from_secs(10);
    const PIPE_ACCESS_OUTBOUND: u32 = 2;
    const PIPE_WAIT: u32 = 0;
    const ERROR_PIPE_CONNECTED: u32 = 535;

    fn wide_zero(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// A named pipe stands in for a slow disk: a read from it stays in flight until the test writes.
    /// Only the CPU backend can open it (PC DirectStorage opens regular files only).
    struct Pipe {
        path: Vec<u16>,
        server: HANDLE,
    }

    impl Pipe {
        fn new(tag: &str) -> Pipe {
            let name = format!("\\\\.\\pipe\\decant_storage_test_{}_{tag}", std::process::id());
            let path = wide_zero(&name);
            // SAFETY: a zero terminated name, the other arguments are plain values.
            let server = unsafe { CreateNamedPipeW(path.as_ptr(), PIPE_ACCESS_OUTBOUND, PIPE_WAIT, 1, 4096, 4096, 0, std::ptr::null()) };
            Pipe { path: path[..path.len() - 1].to_vec(), server }
        }

        fn connect(&self) -> bool {
            // SAFETY: the pipe handle is owned by `self`.
            unsafe { ConnectNamedPipe(self.server, std::ptr::null_mut()) != 0 || windows_sys::Win32::Foundation::GetLastError() == ERROR_PIPE_CONNECTED }
        }

        fn write(&self, data: &[u8]) -> bool {
            let mut done = 0u32;
            // SAFETY: the pipe handle is owned by `self`, `data` is readable.
            unsafe { WriteFile(self.server, data.as_ptr(), data.len() as u32, &mut done, std::ptr::null_mut()) != 0 && done as usize == data.len() }
        }
    }

    // SAFETY: a pipe handle may be used from the writer thread of the test.
    unsafe impl Sync for Pipe {}

    impl Drop for Pipe {
        fn drop(&mut self) {
            if self.server != INVALID_HANDLE_VALUE {
                // SAFETY: the handle is owned by `self`.
                unsafe { CloseHandle(self.server) };
            }
        }
    }

    fn read(file: &Obj, offset: u64, size: u32, destination: *mut c_void, destination_size: u32, codec: u32) -> DecantRead {
        DecantRead { file: file.ptr(), offset, source_size: size, destination, destination_size, codec, bc_mode: 0 }
    }

    /// Asynchronous submission, several queues at once, status order, and closing a queue during a read.
    fn queue_behaviour(f: &Obj, regular_file: &[u16], raw: &[u8]) -> u8 {
        let payload: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let dstorage = wide_zero("dstorage.dll");
        // SAFETY: a zero terminated name.
        if unsafe { !GetModuleHandleW(dstorage.as_ptr()).is_null() } {
            // PC DirectStorage reads files by size: a pipe has none, so the slow disk stand-in cannot work.
            println!("SKIP: queue behaviour tests need the CPU backend (PC DirectStorage is loaded)");
            return 0;
        }
        {
            let pipe = Pipe::new("async");
            check!(pipe.server != INVALID_HANDLE_VALUE);
            let Ok(slow) = f.open_file(&pipe.path) else { return fail(line!(), "SUCCEEDED(f->OpenFile(pipe.path.c_str(), &slow))") };
            check!(pipe.connect());
            let (Ok(fast), Ok(a), Ok(b)) = (f.open_file(regular_file), f.create_queue(32), f.create_queue(32)) else {
                return fail(line!(), "the fast file and the two queues open");
            };
            let (Ok(sa), Ok(sb)) = (f.create_status_array(1), f.create_status_array(2)) else {
                return fail(line!(), "the two status arrays are created");
            };

            // Queue a: one read held by the pipe. Submit returns while the read is still in flight.
            let mut got = [0u8; 16];
            check!(a.enqueue(&read(&slow, 0, 16, got.as_mut_ptr().cast(), 16, codec::COPY)) >= 0);
            check!(a.enqueue_status(&sa, 0) >= 0);
            check!(a.submit() == S_OK);
            check!(sa.result(0) == E_PENDING);

            // Queue b reads a regular file meanwhile and completes: queues run in parallel. Its two
            // statuses complete in enqueue order.
            let mut x = vec![0u8; 4096];
            let mut y = vec![0u8; 4096];
            check!(b.enqueue(&read(&fast, 0, 4096, x.as_mut_ptr().cast(), 4096, codec::COPY)) >= 0);
            check!(b.enqueue_status(&sb, 0) >= 0);
            check!(b.enqueue(&read(&fast, 4096, 4096, y.as_mut_ptr().cast(), 4096, codec::COPY)) >= 0);
            check!(b.enqueue_status(&sb, 1) >= 0);
            check!(b.submit() == S_OK);
            let mut ordered = true;
            check!(wait_for(
                || {
                    let (first, second) = (sb.result(0), sb.result(1));
                    if second != E_PENDING && first == E_PENDING {
                        ordered = false;
                    }
                    second != E_PENDING
                },
                TIMEOUT
            ));
            check!(ordered && sb.result(0) == S_OK && sb.result(1) == S_OK);
            check!(x == raw[..4096] && y == raw[4096..8192]);
            check!(sa.result(0) == E_PENDING); // queue a is still waiting on its pipe

            check!(pipe.write(&payload));
            check!(wait_status_default(&sa, 0) == S_OK && got == payload);
        }
        {
            // Closing a queue during a read: the read in flight finishes, everything after it is cancelled,
            // and Release returns only once nothing is left pending.
            let pipe = Pipe::new("close");
            check!(pipe.server != INVALID_HANDLE_VALUE);
            let Ok(slow) = f.open_file(&pipe.path) else { return fail(line!(), "SUCCEEDED(f->OpenFile(pipe.path.c_str(), &slow))") };
            check!(pipe.connect());
            let Ok(fast) = f.open_file(regular_file) else { return fail(line!(), "SUCCEEDED(f->OpenFile(regularFile, &fast))") };
            let (Ok(mut q), Ok(s)) = (f.create_queue(32), f.create_status_array(2)) else {
                return fail(line!(), "SUCCEEDED(f->CreateQueue(32, &q)) && SUCCEEDED(f->CreateStatusArray(2, &s))");
            };
            let mut got = [0u8; 16];
            let mut after = vec![0xCDu8; 64];
            // SAFETY: plain Win32 call.
            let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
            check!(!event.is_null());
            check!(q.enqueue(&read(&slow, 0, 16, got.as_mut_ptr().cast(), 16, codec::COPY)) >= 0);
            check!(q.enqueue_status(&s, 0) >= 0);
            check!(q.enqueue(&read(&fast, 0, 64, after.as_mut_ptr().cast(), 64, codec::COPY)) >= 0);
            check!(q.enqueue_status(&s, 1) >= 0);
            check!(q.enqueue_event(event) >= 0);
            check!(q.submit() == S_OK);
            std::thread::sleep(Duration::from_millis(200)); // let the worker start the pipe read, so it is really in flight when the queue closes
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    std::thread::sleep(Duration::from_millis(300));
                    pipe.write(&payload);
                });
                q.reset(); // blocks until the pipe read completes
            });
            check!(got == payload); // the read in flight completed
            check!(s.result(0) == hresult_from_win32(ERROR_CANCELLED));
            check!(s.result(1) == hresult_from_win32(ERROR_CANCELLED));
            check!(after.iter().all(|&v| v == 0xCD));
            // SAFETY: the event is ours.
            check!(unsafe { WaitForSingleObject(event, 0) } == WAIT_TIMEOUT);
            // SAFETY: closed once.
            unsafe { CloseHandle(event) };
        }
        println!("PASS: asynchronous Submit, parallel queues, status order, close during a read");
        0
    }

    fn fail(line: u32, what: &str) -> u8 {
        eprintln!("FAIL line {line}: {what}");
        1
    }

    pub fn run(dll: &Dll) -> u8 {
        let fence = TestFence::new();
        let Some(temp) = TempFile::new("sds") else { return fail(line!(), "GetTempPathW(MAX_PATH, directory) && GetTempFileNameW(directory, L\"sds\", 0, temp.path)") };
        let raw: Vec<u8> = (0..70000usize).map(|i| (i * 13 + (i >> 7)) as u8).collect();
        let Some(z) = zlib_compress(&raw, 1) else { return fail(line!(), "zn") };
        let Some(g) = gdeflate_compress(&raw, 1) else { return fail(line!(), "GDeflate::Compress(g.data(), &gn, raw.data(), raw.size(), 1, GDeflate::COMPRESS_SINGLE_THREAD)") };
        let (zn, gn) = (z.len(), g.len());
        let contents: Vec<u8> = [raw.as_slice(), z.as_slice(), g.as_slice()].concat();
        check!(std::fs::write(&temp.path, contents).is_ok());
        let mut unknown = std::ptr::dangling_mut::<c_void>();
        let null_guid = GUID::from_u128(0);
        check!(dll.get_factory_raw(&null_guid, &mut unknown) == E_NOINTERFACE && unknown.is_null());
        let (hr, f) = dll.get_factory(&IID_DECANT_FACTORY);
        check!(hr >= 0);
        let (Ok(file), Ok(mut queue), Ok(status)) = (f.open_file(&temp.wide()), f.create_queue(32), f.create_status_array(2)) else {
            return fail(line!(), "SUCCEEDED(f->OpenFile(temp.path, &file)) && SUCCEEDED(f->CreateQueue(32, &queue)) && SUCCEEDED(f->CreateStatusArray(2, &status))");
        };
        let mut slice = [0u8; 3];
        let mut a = vec![0u8; raw.len()];
        let mut b = vec![0u8; raw.len()];
        let mut console = DecantConsoleRead::zeroed();
        console.file = file.ptr();
        console.offset = 107;
        console.source_size = 3;
        console.destination = slice.as_mut_ptr().cast();
        console.destination_size = 3;
        console.options = 1 << 5; // a swizzle mode, not implemented
        check!(queue.enqueue_console(&console) == E_NOTIMPL);
        console.options = 0;
        check!(queue.enqueue_console(&console) >= 0);
        let mut compressed = console;
        compressed.options = 1;
        compressed.offset = raw.len() as u64;
        compressed.destination = a.as_mut_ptr().cast();
        compressed.destination_size = a.len() as u32;
        compressed.source_size = zn as u32;
        check!(queue.enqueue_console(&compressed) >= 0);
        check!(queue.enqueue(&read(&file, (raw.len() + zn) as u64, gn as u32, b.as_mut_ptr().cast(), b.len() as u32, codec::GDEFLATE)) >= 0);
        check!(queue.enqueue_status(&status, 0) >= 0);
        // SAFETY: plain Win32 call.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        check!(!event.is_null() && queue.enqueue_event(event) >= 0);
        check!(queue.enqueue_fence(&fence, 77) >= 0 && fence.value.load(std::sync::atomic::Ordering::SeqCst) == 0);
        // SAFETY: the event is ours.
        check!(unsafe { WaitForSingleObject(event, 0) } == WAIT_TIMEOUT && status.result(0) == E_PENDING);
        check!(queue.submit() == S_OK);
        // SAFETY: the event is ours.
        check!(unsafe { WaitForSingleObject(event, 10000) } == WAIT_OBJECT_0);
        check!(wait_for(|| fence.value.load(std::sync::atomic::Ordering::SeqCst) == 77 && fence.refs.load(std::sync::atomic::Ordering::SeqCst) == 1, TIMEOUT));
        // SAFETY: closed once.
        unsafe { CloseHandle(event) };
        check!(status.result(0) == S_OK && a == raw && b == raw && slice == raw[107..110]);
        let previous = slice;
        console.options = 2;
        console.offset = 0;
        check!(queue.enqueue_console(&console) >= 0);
        check!(queue.enqueue_status(&status, 1) >= 0);
        // Three bytes are no BC1 BCPack chunk: refused, nothing written.
        check!(queue.submit() == S_OK && wait_status_default(&status, 1) < 0 && slice == previous);
        check!(queue.enqueue(&read(&file, 0, 3, slice.as_mut_ptr().cast(), 3, codec::ZLIB)) >= 0);
        check!(queue.enqueue_status(&status, 1) >= 0);
        check!(queue.submit() == S_OK && wait_status_default(&status, 1) < 0);
        check!(queue.enqueue(&read(&file, 2, 3, slice.as_mut_ptr().cast(), 3, codec::COPY)) >= 0);
        check!(queue.enqueue_status(&status, 1) >= 0);
        check!(queue.submit() == S_OK && wait_status_default(&status, 1) == S_OK);
        check!(queue.enqueue(&read(&file, (raw.len() + zn + gn) as u64, 3, slice.as_mut_ptr().cast(), 3, codec::COPY)) >= 0);
        check!(queue.enqueue_status(&status, 0) >= 0);
        check!(queue.submit() == S_OK && wait_status_default(&status, 0) < 0 && status.result(0) != hresult_from_win32(WAIT_TIMEOUT));
        check!(queue.enqueue_status(&status, 1) >= 0);
        queue.reset();
        check!(status.result(1) == hresult_from_win32(ERROR_CANCELLED));
        println!("PASS: queued reads, console layout, zlib, two-tile GDeflate, status, event, fence, BCPack error, EOF and recovery");

        queue_behaviour(&f, &temp.wide(), &raw)
    }
}
