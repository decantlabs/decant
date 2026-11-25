//! PC DirectStorage (`dstorage.dll`), loaded at run time the way the C++ loaded it: nothing of
//! the SDK is needed to build. The interfaces are declared here by their vtables, in the order of
//! the public header `dstorage.h` (Microsoft.Direct3D.DirectStorage 1.3.0), and only the methods
//! the engine calls are named.
//!
//! `DECANT_DSTORAGE_PC_DLL` picks the runtime: `cpu` keeps the Win32 CPU backend, a path loads that
//! DLL (a failure is an error), unset looks for `dstorage.dll` in the application folder and
//! falls back to the CPU backend when it is missing or does not give a factory.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::NonNull;

use windows_sys::Win32::Foundation::{FreeLibrary, GetLastError, HANDLE, HMODULE};
use windows_sys::Win32::Storage::FileSystem::BY_HANDLE_FILE_INFORMATION;
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_APPLICATION_DIR, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};

use crate::log::storage_log;
use crate::decant::{ERROR_MOD_NOT_FOUND, ERROR_PROC_NOT_FOUND, GUID, HRESULT, S_OK, UnknownVtbl, E_INVALIDARG, hresult_from_win32, release};

const IID_FACTORY: GUID = GUID::from_u128(0x6924ea0c_c3cd_4826_b10a_f64f4ed927c1);
pub const IID_FILE: GUID = GUID::from_u128(0x5de95e7b_955a_4868_a73c_243b29f4b8da);
pub const IID_QUEUE1: GUID = GUID::from_u128(0xdd2f482c_5eff_41e8_9c9e_d2374b278128);
pub const IID_STATUS_ARRAY: GUID = GUID::from_u128(0x82397587_7cd5_453b_a02e_31379bd64656);

/// `DSTORAGE_MIN_QUEUE_CAPACITY`.
pub const MIN_QUEUE_CAPACITY: u16 = 0x80;
/// `DSTORAGE_COMPRESSION_FORMAT_GDEFLATE`.
pub const COMPRESSION_GDEFLATE: u8 = 1;

/// `DSTORAGE_QUEUE_DESC`: source type 0 (file), capacity, priority, name, device.
#[repr(C)]
pub struct QueueDesc {
    pub source_type: u64,
    pub capacity: u16,
    pub priority: i8,
    pub name: *const u8,
    pub device: *mut c_void,
}

const _: () = assert!(size_of::<QueueDesc>() == 32);

/// `DSTORAGE_REQUEST` for a file source and a memory destination, the only kind the engine
/// makes, with the unions laid out as their widest member: 104 bytes, options 0, source 16,
/// destination 40, uncompressed size 80, cancellation tag 88, name 96. The options are the
/// compression format byte, seven reserved bytes, then the source and destination type bits (file
/// and memory, both 0).
#[repr(C)]
pub struct RawRequest {
    compression: u8,
    reserved1: [u8; 7],
    options_bits: u64,
    source_file: *mut c_void,
    source_offset: u64,
    source_size: u32,
    _source_padding: u32,
    destination_buffer: *mut c_void,
    destination_size: u32,
    _destination_union: [u8; 28],
    uncompressed_size: u32,
    _padding: u32,
    cancellation_tag: u64,
    name: *const u8,
}

const _: () = assert!(size_of::<RawRequest>() == 104);
const _: () = assert!(std::mem::offset_of!(RawRequest, source_file) == 16 && std::mem::offset_of!(RawRequest, destination_buffer) == 40);
const _: () = assert!(std::mem::offset_of!(RawRequest, uncompressed_size) == 80 && std::mem::offset_of!(RawRequest, cancellation_tag) == 88);

impl RawRequest {
    /// A file to memory request, compressed with `compression` (0 none, 1 GDeflate).
    pub fn file_to_memory(compression: u8, file: *mut c_void, offset: u64, source_size: u32, buffer: *mut c_void, buffer_size: u32) -> RawRequest {
        RawRequest {
            compression,
            reserved1: [0; 7],
            options_bits: 0,
            source_file: file,
            source_offset: offset,
            source_size,
            _source_padding: 0,
            destination_buffer: buffer,
            destination_size: buffer_size,
            _destination_union: [0; 28],
            uncompressed_size: buffer_size,
            _padding: 0,
            cancellation_tag: 0,
            name: std::ptr::null(),
        }
    }
}

#[repr(C)]
pub struct FactoryVtbl {
    pub base: UnknownVtbl,
    pub create_queue: unsafe extern "system" fn(*mut c_void, *const QueueDesc, *const GUID, *mut *mut c_void) -> HRESULT,
    pub open_file: unsafe extern "system" fn(*mut c_void, *const u16, *const GUID, *mut *mut c_void) -> HRESULT,
    pub create_status_array: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT,
}

#[repr(C)]
pub struct FileVtbl {
    pub base: UnknownVtbl,
    pub close: unsafe extern "system" fn(*mut c_void),
    pub get_file_information: unsafe extern "system" fn(*mut c_void, *mut BY_HANDLE_FILE_INFORMATION) -> HRESULT,
}

/// `IDStorageQueue1`.
#[repr(C)]
pub struct QueueVtbl {
    pub base: UnknownVtbl,
    pub enqueue_request: unsafe extern "system" fn(*mut c_void, *const RawRequest),
    pub enqueue_status: unsafe extern "system" fn(*mut c_void, *mut c_void, u32),
    pub enqueue_signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64),
    pub submit: unsafe extern "system" fn(*mut c_void),
    pub cancel_requests_with_tag: unsafe extern "system" fn(*mut c_void, u64, u64),
    pub close: unsafe extern "system" fn(*mut c_void),
    pub get_error_event: unsafe extern "system" fn(*mut c_void) -> HANDLE,
    pub retrieve_error_record: unsafe extern "system" fn(*mut c_void, *mut c_void),
    pub query: unsafe extern "system" fn(*mut c_void, *mut c_void),
    pub enqueue_set_event: unsafe extern "system" fn(*mut c_void, HANDLE),
}

#[repr(C)]
pub struct StatusArrayVtbl {
    pub base: UnknownVtbl,
    pub is_complete: unsafe extern "system" fn(*mut c_void, u32) -> bool,
    pub get_hresult: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT,
}

/// An owned interface pointer of the PC runtime (the `ComPtr` of the C++).
pub struct Pc(NonNull<c_void>);

// SAFETY: DirectStorage objects are free threaded: the C++ used them from the queue's worker
// thread after creating them on the caller's.
unsafe impl Send for Pc {}
// SAFETY: as above.
unsafe impl Sync for Pc {}

impl Pc {
    /// Takes ownership of an interface pointer, which may be null.
    ///
    /// # Safety
    /// `raw` is null or an interface pointer whose reference the caller gives up.
    pub unsafe fn from_raw(raw: *mut c_void) -> Option<Pc> {
        NonNull::new(raw).map(Pc)
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    /// The vtable, as the interface `V` the pointer was asked for.
    ///
    /// # Safety
    /// The object must be the interface `V` describes.
    pub unsafe fn vtbl<V>(&self) -> &V {
        // SAFETY: the caller states the interface.
        unsafe { crate::decant::vtable::<V>(self.as_ptr()) }
    }
}

impl Drop for Pc {
    fn drop(&mut self) {
        // SAFETY: gives back the reference this value owns.
        unsafe { release(self.as_ptr()) };
    }
}

/// What the engine runs on: the PC runtime when it loaded, else nothing (the Win32 CPU backend).
pub struct Backend {
    module: HMODULE,
    pub factory: Option<Pc>,
    /// Why the backend cannot be used, `S_OK` when it can.
    pub hr: HRESULT,
}

// SAFETY: a module handle is a plain address, and the factory is free threaded.
unsafe impl Send for Backend {}
// SAFETY: as above.
unsafe impl Sync for Backend {}

/// The value of `DECANT_DSTORAGE_PC_DLL`, `None` when unset or empty. At 32768 UTF-16 units or more
/// the C++ reader refused it, which `Err` reports.
fn runtime_override() -> Result<Option<std::ffi::OsString>, ()> {
    use std::os::windows::ffi::OsStrExt;
    match std::env::var_os("DECANT_DSTORAGE_PC_DLL").filter(|v| !v.is_empty()) {
        Some(v) if v.encode_wide().count() >= 32768 => Err(()),
        other => Ok(other),
    }
}

impl Backend {
    /// The Win32 CPU backend, whatever the environment says.
    #[cfg(test)]
    pub fn cpu() -> Backend {
        Backend { module: std::ptr::null_mut(), factory: None, hr: S_OK }
    }

    pub fn new() -> Backend {
        let mut backend = Backend { module: std::ptr::null_mut(), factory: None, hr: S_OK };
        let path = match runtime_override() {
            Ok(path) => path,
            Err(()) => {
                backend.hr = E_INVALIDARG;
                return backend;
            }
        };
        if path.as_deref().is_some_and(|p| p == "cpu") {
            return backend;
        }
        use std::os::windows::ffi::OsStrExt;
        let name: Vec<u16> = match &path {
            Some(p) => p.encode_wide().chain(std::iter::once(0)).collect(),
            None => "dstorage.dll".encode_utf16().chain(std::iter::once(0)).collect(),
        };
        let flags = if path.is_some() { LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 } else { LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 };
        // SAFETY: `name` is a zero terminated path that lives for the call.
        backend.module = unsafe { LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), flags) };
        if backend.module.is_null() {
            // SAFETY: no precondition.
            let error = unsafe { GetLastError() };
            if path.is_some() || error != ERROR_MOD_NOT_FOUND {
                backend.hr = hresult_from_win32(error);
            }
            return backend;
        }
        type GetFactory = unsafe extern "system" fn(*const GUID, *mut *mut c_void) -> HRESULT;
        // SAFETY: the module is loaded and the name is zero terminated.
        let get = unsafe { GetProcAddress(backend.module, c"DStorageGetFactory".as_ptr().cast()) };
        backend.hr = match get {
            Some(get) => {
                let mut raw = std::ptr::null_mut();
                // SAFETY: the export has the signature of `DStorageGetFactory`.
                let hr = unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, GetFactory>(get)(&IID_FACTORY, &mut raw) };
                // SAFETY: on success `raw` is a factory reference we now own.
                backend.factory = unsafe { Pc::from_raw(raw) };
                hr
            }
            None => hresult_from_win32(ERROR_PROC_NOT_FOUND),
        };
        if path.is_none() && backend.hr < 0 {
            storage_log(&format!("PC DStorageGetFactory unavailable hr={:08x}; using Win32 CPU backend\n", backend.hr as u32));
            backend.factory = None;
            backend.hr = S_OK;
        }
        backend
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.factory = None;
        if !self.module.is_null() {
            // SAFETY: the module was loaded by `new` and is freed once, after its factory.
            unsafe { FreeLibrary(self.module) };
        }
    }
}
