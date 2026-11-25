//! Windows calls of the smoke test: finding and loading DLLs, and calling the exports it checks
//! through the signatures of the C version's typedefs. Calling into a DLL found at run time is
//! `unsafe` by nature, so it is confined to this module.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_void};
use std::os::windows::ffi::OsStrExt;

use windows_sys::Win32::Foundation::{E_FAIL, FARPROC, GetLastError, HANDLE, HMODULE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{FindClose, FindFirstFileW, FindNextFileW, WIN32_FIND_DATAW};
use windows_sys::Win32::System::Diagnostics::Debug::{SEM_FAILCRITICALERRORS, SEM_NOOPENFILEERRORBOX, SetErrorMode};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::HRESULT;

/// What `GetProcAddress` answers, before the cast to the export's own signature.
type Raw = unsafe extern "system" fn() -> isize;

fn wide(text: &str) -> Vec<u16> {
    std::ffi::OsStr::new(text).encode_wide().chain([0]).collect()
}

/// No message box when a DLL or a drive is missing: failures are reported as RESULT lines.
pub fn quiet_errors() {
    // SAFETY: plain process flag.
    unsafe { SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX) };
}

/// `FindFirstFile("*.dll")` in the working directory, in the order the file system answers.
/// `None` when nothing matches.
pub fn local_dlls() -> Option<Vec<String>> {
    let pattern = wide("*.dll");
    // SAFETY: plain structure the calls fill.
    let mut data: WIN32_FIND_DATAW = unsafe { std::mem::zeroed() };
    // SAFETY: NUL terminated pattern, `data` lives for the whole search.
    let find = unsafe { FindFirstFileW(pattern.as_ptr(), &mut data) };
    if find == INVALID_HANDLE_VALUE {
        return None;
    }
    let mut names = Vec::new();
    loop {
        let name = &data.cFileName;
        let len = name.iter().position(|&c| c == 0).unwrap_or(name.len());
        names.push(String::from_utf16_lossy(&name[..len]));
        // SAFETY: the handle comes from FindFirstFileW above.
        if unsafe { FindNextFileW(find, &mut data) } == 0 {
            break;
        }
    }
    // SAFETY: closes the handle opened above, once.
    unsafe { FindClose(find) };
    Some(names)
}

/// A loaded module, never freed (the C version kept them all loaded too).
#[derive(Clone, Copy)]
pub struct Module(HMODULE);

impl Module {
    /// `LoadLibrary(name)`, or the error code of the failure.
    pub fn load(name: &str) -> Result<Self, u32> {
        let name = wide(name);
        // SAFETY: NUL terminated name. Loading runs the DLL's entry point: that is the test.
        let module = unsafe { LoadLibraryW(name.as_ptr()) };
        if module.is_null() {
            // SAFETY: plain call, right after the failure.
            Err(unsafe { GetLastError() })
        } else {
            Ok(Self(module))
        }
    }

    /// `GetModuleHandle(name)`: the module if the process has loaded it.
    pub fn loaded(name: &str) -> Option<Self> {
        let name = wide(name);
        // SAFETY: NUL terminated name.
        let module = unsafe { GetModuleHandleW(name.as_ptr()) };
        (!module.is_null()).then_some(Self(module))
    }

    pub fn address(self) -> usize {
        self.0 as usize
    }

    fn proc(self, name: &CStr) -> FARPROC {
        // SAFETY: a live module and a NUL terminated name.
        unsafe { GetProcAddress(self.0, name.as_ptr().cast()) }
    }

    /// The address of an export, resolving a forwarder.
    pub fn export(self, name: &CStr) -> Option<usize> {
        self.proc(name).map(|f| f as usize)
    }
}

type XMemAlloc = unsafe extern "system" fn(usize, u64) -> *mut c_void;
type XMemFree = unsafe extern "system" fn(*mut c_void, u64);
/// The information block is 64 bytes, see the crate decant-xmem, `xmem_abitest`.
type XMemVirtualQuery = unsafe extern "system" fn(HANDLE, *const c_void, *mut c_void) -> i32;

/// The allocator exports of `xmem.dll`.
pub struct XMem {
    alloc: XMemAlloc,
    free: XMemFree,
    query: Option<XMemVirtualQuery>,
}

impl XMem {
    /// `None` unless both `XMemAlloc` and `XMemFree` are exported.
    pub fn find(module: Module) -> Option<Self> {
        // SAFETY: each export is cast to the signature decant-xmem defines for it.
        unsafe {
            Some(Self {
                alloc: std::mem::transmute::<Raw, XMemAlloc>(module.proc(c"XMemAlloc")?),
                free: std::mem::transmute::<Raw, XMemFree>(module.proc(c"XMemFree")?),
                query: module.proc(c"XMemVirtualQuery").map(|f| std::mem::transmute::<Raw, XMemVirtualQuery>(f)),
            })
        }
    }

    pub fn alloc(&self, size: usize, attributes: u64) -> usize {
        // SAFETY: XMemAlloc takes any size and attributes and answers null on failure.
        unsafe { (self.alloc)(size, attributes) as usize }
    }

    pub fn free(&self, address: usize, attributes: u64) {
        // SAFETY: an address XMemAlloc answered (or null, which it ignores) with its attributes.
        unsafe { (self.free)(address as *mut c_void, attributes) }
    }

    /// Writes `byte` over the whole block, as the C version's `memset`.
    pub fn fill(&self, address: usize, byte: u8, len: usize) {
        // SAFETY: `address` is a live block of at least `len` bytes from `alloc`.
        unsafe { (address as *mut u8).write_bytes(byte, len) }
    }

    pub fn has_query(&self) -> bool {
        self.query.is_some()
    }

    /// `XMemVirtualQuery(GetCurrentProcess(), address, info)`: the BOOL and the 64 byte block.
    pub fn query(&self, address: usize) -> Option<(bool, [u8; 64])> {
        let query = self.query?;
        let mut info = [0u8; 64];
        // SAFETY: the information buffer is the 64 bytes the export writes at most.
        let ok = unsafe { query(GetCurrentProcess(), address as *const c_void, info.as_mut_ptr().cast()) };
        Some((ok != 0, info))
    }
}

type HCGetLibVersion = unsafe extern "system" fn(*mut *const c_char) -> HRESULT;
type HCInitialize = unsafe extern "system" fn(*mut c_void) -> HRESULT;
type HCCleanup = unsafe extern "system" fn();

/// The three exports of the HTTP client library the test calls, each optional as in C.
pub struct HttpClient {
    version: Option<HCGetLibVersion>,
    init: Option<HCInitialize>,
    cleanup: Option<HCCleanup>,
}

impl HttpClient {
    pub fn find(module: Module) -> Self {
        // SAFETY: each export is cast to its signature in the public libHttpClient headers.
        unsafe {
            Self {
                version: module.proc(c"HCGetLibVersion").map(|f| std::mem::transmute::<Raw, HCGetLibVersion>(f)),
                init: module.proc(c"HCInitialize").map(|f| std::mem::transmute::<Raw, HCInitialize>(f)),
                cleanup: module.proc(c"HCCleanup").map(|f| std::mem::transmute::<Raw, HCCleanup>(f)),
            }
        }
    }

    /// `HCGetLibVersion(&v)`: the result and the version text, `E_FAIL` without the export.
    pub fn version(&self) -> (HRESULT, Option<String>) {
        let Some(version) = self.version else { return (E_FAIL, None) };
        let mut text: *const c_char = std::ptr::null();
        // SAFETY: the export writes a pointer to a static NUL terminated string.
        let hr = unsafe { version(&mut text) };
        // SAFETY: null or a NUL terminated string owned by the library.
        (hr, (!text.is_null()).then(|| unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()))
    }

    /// `HCInitialize(NULL)`, `E_FAIL` without the export.
    pub fn initialize(&self) -> HRESULT {
        // SAFETY: null arguments are the documented default.
        self.init.map_or(E_FAIL, |init| unsafe { init(std::ptr::null_mut()) })
    }

    pub fn cleanup(&self) {
        if let Some(cleanup) = self.cleanup {
            // SAFETY: called once, after a successful HCInitialize.
            unsafe { cleanup() }
        }
    }
}

/// `curl_version()` of the CURL wrapper (a `cdecl` function, the same convention on x64).
pub fn curl_version(module: Module) -> Option<String> {
    type CurlVersion = unsafe extern "C" fn() -> *const c_char;
    // SAFETY: cast to the signature of curl.h.
    let version = unsafe { std::mem::transmute::<Raw, CurlVersion>(module.proc(c"curl_version")?) };
    // SAFETY: curl_version answers a static NUL terminated string, or null.
    let text = unsafe { version() };
    // SAFETY: checked for null, NUL terminated by contract.
    (!text.is_null()).then(|| unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned())
}
