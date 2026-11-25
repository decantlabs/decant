//! Test helper: a small stand-in title for the debugging switches of `decant_launch::tools` (see
//! tests/tools.rs). The first argument picks what it does:
//!
//! * `calls`: counts one unanswered question three times through `decant_log::calls`, as a shim does.
//! * `rva`: prints the offset of its watched value inside its own image and exits (no debugger needed).
//! * `watch`: writes the watched value 1, 2, 3 from a named thread, reads it once, then exits.
//! * `gpu <file>`: like `wait`, and a second after its start it opens a DXGI factory and its first adapter, as a title does
//!   (what makes the kernel know it as a GPU user, see tests/gentle.rs).
//! * `wait <file>`: a named thread waits on an event until the file `<file>` exists (the test makes it once it has seen
//!   what it looks for), the main thread joins it. A third thread looks for the file and sets the event. At most 300 s.

use std::sync::atomic::{AtomicU64, Ordering};

/// The value `DECANT_WATCH` is pointed at.
static WATCHED: AtomicU64 = AtomicU64::new(0);

#[cfg(windows)]
#[allow(unsafe_code)]
mod sys {
    use std::ffi::c_void;

    #[link(name = "kernel32", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        pub fn CreateEventW(attributes: *const c_void, manual: i32, initial: i32, name: *const u16) -> *mut c_void;
        pub fn WaitForSingleObject(handle: *mut c_void, ms: u32) -> u32;
        pub fn SetThreadDescription(thread: *mut c_void, name: *const u16) -> i32;
        pub fn GetCurrentThread() -> *mut c_void;
        pub fn SetEvent(handle: *mut c_void) -> i32;
    }

    #[link(name = "dxgi", kind = "raw-dylib")]
    unsafe extern "system" {
        fn CreateDXGIFactory1(riid: *const [u8; 16], factory: *mut *mut c_void) -> i32;
    }

    /// Creates a DXGI factory and asks for its first adapter (`IDXGIFactory1::EnumAdapters1`, vtable slot 12).
    pub fn use_gpu() {
        // IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
        let iid: [u8; 16] = [0x78, 0xae, 0x0a, 0x77, 0x6f, 0xf2, 0xba, 0x4d, 0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87];
        let mut factory: *mut c_void = std::ptr::null_mut();
        // SAFETY: a COM call on the interface it returned: the vtable is read at the slot of EnumAdapters1.
        unsafe {
            if CreateDXGIFactory1(&iid, &mut factory) < 0 || factory.is_null() {
                return;
            }
            let vtable = *(factory as *const *const usize);
            let enum_adapters1: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32 = std::mem::transmute(*vtable.add(12));
            let mut adapter: *mut c_void = std::ptr::null_mut();
            enum_adapters1(factory, 0, &mut adapter);
        }
    }

    pub fn image_base() -> usize {
        // SAFETY: a null name asks for the executable's own module.
        unsafe { GetModuleHandleW(std::ptr::null()) as usize }
    }

    pub fn name_this_thread(name: &str) {
        let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
        // SAFETY: a NUL terminated name for the calling thread.
        unsafe { SetThreadDescription(GetCurrentThread(), wide.as_ptr()) };
    }

    /// An unnamed auto reset event, not set, announced as one of ours.
    pub fn new_event() -> usize {
        // SAFETY: an unnamed event with default security.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) } as usize;
        decant_log::handles::created(event, "probe event");
        event
    }

    /// Waits on `event` without a timeout.
    pub fn wait(event: usize) {
        // SAFETY: a live event handle of this process.
        unsafe { WaitForSingleObject(event as *mut c_void, u32::MAX) };
    }

    pub fn set(event: usize) {
        // SAFETY: a live event handle of this process.
        unsafe { SetEvent(event as *mut c_void) };
    }
}

#[cfg(windows)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("calls") => {
            for _ in 0..3 {
                decant_log::calls::not_implemented("probe question");
            }
            decant_log::calls::stub("probe stub");
        }
        Some("rva") => println!("{:#x}", std::ptr::from_ref(&WATCHED) as usize - sys::image_base()),
        Some("watch") => {
            std::thread::Builder::new()
                .name("writer".into())
                .spawn(|| {
                    sys::name_this_thread("probe writer");
                    for v in 1..=3 {
                        WATCHED.store(v, Ordering::SeqCst);
                    }
                })
                .unwrap()
                .join()
                .unwrap();
            std::process::exit(WATCHED.load(Ordering::SeqCst) as i32);
        }
        Some("wait" | "gpu") => {
            let done = std::path::PathBuf::from(args.get(2).cloned().unwrap_or_default());
            let event = sys::new_event();
            if args[1] == "gpu" {
                std::thread::sleep(std::time::Duration::from_secs(1));
                sys::use_gpu();
            }
            let waiter = std::thread::spawn(move || {
                sys::name_this_thread("probe waiter");
                sys::wait(event);
            });
            std::thread::spawn(move || {
                sys::name_this_thread("probe release");
                let end = std::time::Instant::now() + std::time::Duration::from_secs(300);
                while !done.exists() && std::time::Instant::now() < end {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                sys::set(event);
            });
            waiter.join().unwrap();
        }
        _ => std::process::exit(2),
    }
}

#[cfg(not(windows))]
fn main() {
    let _ = WATCHED.load(Ordering::SeqCst);
}
