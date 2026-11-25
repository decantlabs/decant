//! What the two test programs share: the DLL loaded by path, and its objects called through their
//! raw vtables, the way a title (or the C++ tests) called them.
#![allow(unsafe_code)]
#![allow(dead_code)] // each test program uses part of it

use std::ffi::{OsStr, c_void};
use std::os::windows::ffi::OsStrExt;
use std::time::{Duration, Instant};

use dstorage_xs::decant::*;
use decant_testkit::Library;
use windows_sys::Win32::Foundation::HANDLE;

/// A line of the C tests: `FAIL line N: expression` on standard error, and the program returns 1.
#[macro_export]
macro_rules! check {
    ($e:expr) => {
        if !($e) {
            eprintln!("FAIL line {}: {}", line!(), stringify!($e));
            return 1;
        }
    };
}

pub type GetFactory = unsafe extern "system" fn(*const GUID, *mut *mut c_void) -> HRESULT;

/// The DLL under test and its one export.
pub struct Dll {
    _library: Library,
    get_factory: GetFactory,
}

impl Dll {
    pub fn load(path: &OsStr) -> Option<Dll> {
        let library = Library::open(path).ok()?;
        // SAFETY: `DStorageGetFactory` has this signature.
        let get_factory = unsafe { library.get::<GetFactory>("DStorageGetFactory") }?;
        Some(Dll { _library: library, get_factory })
    }

    /// `DStorageGetFactory(iid, &mut out)`: the result and the object, null on failure.
    pub fn get_factory(&self, iid: &GUID) -> (HRESULT, Obj) {
        let mut out = std::ptr::null_mut();
        // SAFETY: both pointers are valid for the call.
        let hr = unsafe { (self.get_factory)(iid, &mut out) };
        (hr, Obj(out))
    }

    /// The same, with an output that starts as a non null pointer, to see it cleared.
    pub fn get_factory_raw(&self, iid: &GUID, out: &mut *mut c_void) -> HRESULT {
        // SAFETY: both pointers are valid for the call.
        unsafe { (self.get_factory)(iid, out) }
    }
}

/// An owned COM reference, released on drop.
pub struct Obj(pub *mut c_void);

impl Obj {
    pub fn null() -> Obj {
        Obj(std::ptr::null_mut())
    }

    pub fn ptr(&self) -> *mut c_void {
        self.0
    }

    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Releases the reference now (`Reset()` of the C++ `ComPtr`).
    pub fn reset(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the reference is owned by this value.
            unsafe { release(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }

    /// Gives the reference to the caller.
    pub fn take(mut self) -> *mut c_void {
        std::mem::replace(&mut self.0, std::ptr::null_mut())
    }

    // The project factory.
    pub fn open_file(&self, path: &[u16]) -> Result<Obj, HRESULT> {
        let mut zero_terminated = path.to_vec();
        zero_terminated.push(0);
        let mut out = std::ptr::null_mut();
        // SAFETY: a factory of the project ABI, a zero terminated path and a writable output.
        let hr = unsafe { (vtable::<FactoryVtbl>(self.0).open_file)(self.0, zero_terminated.as_ptr(), &mut out) };
        if hr < 0 { Err(hr) } else { Ok(Obj(out)) }
    }

    pub fn create_queue(&self, capacity: u16) -> Result<Obj, HRESULT> {
        let mut out = std::ptr::null_mut();
        // SAFETY: a factory of the project ABI and a writable output.
        let hr = unsafe { (vtable::<FactoryVtbl>(self.0).create_queue)(self.0, capacity, &mut out) };
        if hr < 0 { Err(hr) } else { Ok(Obj(out)) }
    }

    pub fn create_status_array(&self, count: u32) -> Result<Obj, HRESULT> {
        let mut out = std::ptr::null_mut();
        // SAFETY: a factory of the project ABI and a writable output.
        let hr = unsafe { (vtable::<FactoryVtbl>(self.0).create_status_array)(self.0, count, &mut out) };
        if hr < 0 { Err(hr) } else { Ok(Obj(out)) }
    }

    // The project queue.
    pub fn enqueue(&self, read: &DecantRead) -> HRESULT {
        // SAFETY: a queue of the project ABI and a readable request.
        unsafe { (vtable::<QueueVtbl>(self.0).enqueue)(self.0, read) }
    }

    pub fn enqueue_status(&self, status: &Obj, index: u32) -> HRESULT {
        // SAFETY: a queue and a status array of the project ABI.
        unsafe { (vtable::<QueueVtbl>(self.0).enqueue_status)(self.0, status.0, index) }
    }

    pub fn enqueue_event(&self, event: HANDLE) -> HRESULT {
        // SAFETY: a queue of the project ABI.
        unsafe { (vtable::<QueueVtbl>(self.0).enqueue_event)(self.0, event) }
    }

    pub fn enqueue_fence(&self, fence: &TestFence, value: u64) -> HRESULT {
        // SAFETY: a queue of the project ABI and a fence that outlives the call.
        unsafe { (vtable::<QueueVtbl>(self.0).enqueue_fence)(self.0, fence.as_ptr(), value) }
    }

    pub fn submit(&self) -> HRESULT {
        // SAFETY: a queue of the project ABI.
        unsafe { (vtable::<QueueVtbl>(self.0).submit)(self.0) }
    }

    pub fn enqueue_console(&self, read: &DecantConsoleRead) -> HRESULT {
        // SAFETY: a queue of the project ABI and a readable request.
        unsafe { (vtable::<QueueVtbl>(self.0).enqueue_console)(self.0, read) }
    }

    // The project status array.
    pub fn result(&self, index: u32) -> HRESULT {
        // SAFETY: a status array of the project ABI.
        unsafe { (vtable::<StatusVtbl>(self.0).result)(self.0, index) }
    }
}

impl Drop for Obj {
    fn drop(&mut self) {
        self.reset();
    }
}

/// A path as the UTF-16 units the DLL takes.
pub fn wide(path: &OsStr) -> Vec<u16> {
    path.encode_wide().collect()
}

/// A CPU only fence probe: the three methods the engine calls, in the slots of `ID3D12Fence`
/// (`GetCompletedValue` 8, `SetEventOnCompletion` 9, `Signal` 10). No D3D device is created.
#[repr(C)]
pub struct TestFence {
    vtbl: &'static [usize; 11],
    pub refs: std::sync::atomic::AtomicU32,
    pub value: std::sync::atomic::AtomicU64,
}

unsafe extern "system" fn fence_query_interface(_: *mut TestFence, _: *const GUID, _: *mut *mut c_void) -> HRESULT {
    E_NOINTERFACE
}
unsafe extern "system" fn fence_add_ref(this: *mut TestFence) -> u32 {
    // SAFETY: called through the vtable of a `TestFence`.
    unsafe { (*this).refs.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1 }
}
unsafe extern "system" fn fence_release(this: *mut TestFence) -> u32 {
    // SAFETY: called through the vtable of a `TestFence`.
    unsafe { (*this).refs.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) - 1 }
}
unsafe extern "system" fn fence_not_implemented(_: *mut TestFence) -> HRESULT {
    E_NOTIMPL
}
unsafe extern "system" fn fence_completed_value(this: *mut TestFence) -> u64 {
    // SAFETY: called through the vtable of a `TestFence`.
    unsafe { (*this).value.load(std::sync::atomic::Ordering::SeqCst) }
}
unsafe extern "system" fn fence_signal(this: *mut TestFence, value: u64) -> HRESULT {
    // SAFETY: called through the vtable of a `TestFence`.
    unsafe { (*this).value.store(value, std::sync::atomic::Ordering::SeqCst) };
    S_OK
}

static FENCE_VTBL: std::sync::OnceLock<[usize; 11]> = std::sync::OnceLock::new();

impl TestFence {
    pub fn new() -> Box<TestFence> {
        let vtbl = FENCE_VTBL.get_or_init(|| {
            let not_implemented = fence_not_implemented as *const () as usize;
            [
                fence_query_interface as *const () as usize,
                fence_add_ref as *const () as usize,
                fence_release as *const () as usize,
                not_implemented, // GetPrivateData
                not_implemented, // SetPrivateData
                not_implemented, // SetPrivateDataInterface
                not_implemented, // SetName
                not_implemented, // GetDevice
                fence_completed_value as *const () as usize,
                not_implemented, // SetEventOnCompletion
                fence_signal as *const () as usize,
            ]
        });
        Box::new(TestFence { vtbl, refs: std::sync::atomic::AtomicU32::new(1), value: std::sync::atomic::AtomicU64::new(0) })
    }

    pub fn as_ptr(&self) -> *mut c_void {
        std::ptr::from_ref(self).cast_mut().cast()
    }
}

/// Submit is asynchronous: completion is observed through the status array, with a timeout.
pub fn wait_status(status: &Obj, index: u32, timeout: Duration) -> HRESULT {
    let end = Instant::now() + timeout;
    let mut hr = status.result(index);
    while hr == E_PENDING && Instant::now() < end {
        std::thread::sleep(Duration::from_millis(1));
        hr = status.result(index);
    }
    if hr == E_PENDING { hresult_from_win32(WAIT_TIMEOUT) } else { hr }
}

/// `waitStatus` with the ten second timeout of the C++ test.
pub fn wait_status_default(status: &Obj, index: u32) -> HRESULT {
    wait_status(status, index, Duration::from_secs(10))
}

/// Polls `done` until it holds or the timeout passes.
pub fn wait_for(mut done: impl FnMut() -> bool, timeout: Duration) -> bool {
    let end = Instant::now() + timeout;
    while !done() && Instant::now() < end {
        std::thread::sleep(Duration::from_millis(1));
    }
    done()
}

/// A temporary file, deleted on drop.
pub struct TempFile {
    pub path: std::path::PathBuf,
}

impl TempFile {
    pub fn new(prefix: &str) -> Option<TempFile> {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("{prefix}{}_{n}.tmp", std::process::id()));
        std::fs::File::create(&path).ok()?;
        Some(TempFile { path })
    }

    pub fn wide(&self) -> Vec<u16> {
        wide(self.path.as_os_str())
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The function at `index` of the vtable of any COM object, as the function type `F`.
///
/// # Safety
/// `object` is a COM object and `F` is the function pointer type of the method at `index`.
pub unsafe fn slot<F: Copy>(object: *mut c_void, index: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    // SAFETY: the caller states the object, the slot and the signature.
    unsafe {
        let table = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*table.add(index))
    }
}
