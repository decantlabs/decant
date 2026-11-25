//! The project owned adapter ABI (the former `decant_storage.h`): the four interfaces and the two
//! request structures the console layer is built on, plus the small COM machinery behind the
//! objects of `engine`. Never use these interface IDs as console IIDs.
//!
//! The test programs load the DLL and call it through the same vtable structures, which is why
//! they are public.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

pub use windows_sys::core::GUID;
use windows_sys::Win32::Foundation::HANDLE;

pub type HRESULT = i32;

pub const S_OK: HRESULT = 0;
pub const E_PENDING: HRESULT = 0x8000_000A_u32 as i32;
pub const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
pub const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
pub const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
pub const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
pub const E_INVALIDARG: HRESULT = 0x8007_0057_u32 as i32;

pub const ERROR_MOD_NOT_FOUND: u32 = 126;
pub const ERROR_PROC_NOT_FOUND: u32 = 127;
pub const ERROR_INVALID_DATA: u32 = 13;
pub const ERROR_HANDLE_EOF: u32 = 38;
pub const ERROR_BUSY: u32 = 170;
pub const ERROR_CANCELLED: u32 = 1223;
pub const WAIT_TIMEOUT: u32 = 258;

/// The `HRESULT_FROM_WIN32` macro.
pub const fn hresult_from_win32(error: u32) -> HRESULT {
    if (error as i32) <= 0 { error as i32 } else { ((error & 0xFFFF) | 0x8007_0000) as i32 }
}

/// `HRESULT_FROM_WIN32(GetLastError())`.
pub fn last_error() -> HRESULT {
    // SAFETY: no precondition.
    hresult_from_win32(unsafe { windows_sys::Win32::Foundation::GetLastError() })
}

/// The number a GUID's text spells, the form `spec::Guid` holds.
pub fn guid_value(guid: &GUID) -> u128 {
    (u128::from(guid.data1) << 96) | (u128::from(guid.data2) << 80) | (u128::from(guid.data3) << 64) | u128::from(u64::from_be_bytes(guid.data4))
}

/// A GUID in the form `StringFromGUID2` writes: braces, upper case.
pub fn guid_text(guid: &GUID) -> String {
    let d = &guid.data4;
    format!(
        "{{{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}}}",
        guid.data1, guid.data2, guid.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    )
}

pub const IID_IUNKNOWN: GUID = GUID::from_u128(0x00000000_0000_0000_c000_000000000046);
pub const IID_DECANT_FILE: GUID = GUID::from_u128(0x0908dd6a_7861_4eaf_b5de_7c584de49b38);
pub const IID_DECANT_STATUS: GUID = GUID::from_u128(0x69286a0d_4d8c_4a4f_a339_827795691869);
pub const IID_DECANT_QUEUE: GUID = GUID::from_u128(0x6ffa2316_f23b_4bb2_98c7_3d6efd2b227d);
pub const IID_DECANT_FACTORY: GUID = GUID::from_u128(0xbd498cd0_9410_4fdc_8800_b12bc02eb475);

/// The codecs of `DecantRead::codec`.
pub mod codec {
    pub const COPY: u32 = 0;
    pub const ZLIB: u32 = 1;
    pub const GDEFLATE: u32 = 2;
    pub const BCPACK: u32 = 3;
}

/// Natural x64 layout of the file form of the public Xbox `DSTORAGE_REQUEST`: 72 bytes, the
/// destination BEFORE the source, the reverse of the PC `DSTORAGE_REQUEST`. Offsets confirmed on
/// the call sites of retail titles (facts only): options 0, destination 8, page offset 0x10,
/// destination size 0x18, source file 0x20, source offset 0x28, source size 0x30, intermediate
/// size 0x34, cancellation tag 0x38, name 0x40. Another retail title fills the same offsets (its request builder
/// issues one request per chunk of a file): the page offset as a 16 bit store, the
/// intermediate size, tag and name left 0.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DecantConsoleRead {
    /// The bits of `console_options`.
    pub options: u64,
    /// An address, or with `console_options::DESTINATION_PAGES` an array of 64 bit frame numbers
    /// (as `XMemAllocatePhysicalPages` gives them), one per 64 KiB of destination.
    pub destination: *mut c_void,
    /// With `console_options::DESTINATION_PAGES`, the low 16 bits are where the data starts in the
    /// first page. Unused otherwise.
    pub destination_page_offset: u64,
    pub destination_size: u32,
    /// The resolved project file object.
    pub file: *mut c_void,
    pub offset: u64,
    pub source_size: u32,
    /// Unknown field at 0x34.
    pub intermediate_size: u32,
    pub cancellation_tag: u64,
    pub name: *const u8,
}

const _: () = assert!(size_of::<DecantConsoleRead>() == 72 && std::mem::offset_of!(DecantConsoleRead, destination) == 8);
const _: () = assert!(std::mem::offset_of!(DecantConsoleRead, file) == 32 && std::mem::offset_of!(DecantConsoleRead, cancellation_tag) == 56);

impl DecantConsoleRead {
    pub const fn zeroed() -> DecantConsoleRead {
        DecantConsoleRead {
            options: 0,
            destination: std::ptr::null_mut(),
            destination_page_offset: 0,
            destination_size: 0,
            file: std::ptr::null_mut(),
            offset: 0,
            source_size: 0,
            intermediate_size: 0,
            cancellation_tag: 0,
            name: std::ptr::null(),
        }
    }
}

/// The option bits of a console request (`DecantConsoleRead::options`), in the order of the public Xbox
/// declaration of the request options, and as retail request builders use them.
pub mod console_options {
    /// Inflate zlib (RFC 1950) data.
    pub const ZLIB: u64 = 1;
    /// Bits 1 to 4: unpack BCPack data of this BC format (1 = BC1 ... 7 = BC7), 0 for none.
    pub const BCPACK_SHIFT: u32 = 1;
    pub const BCPACK_MASK: u64 = 0xF << BCPACK_SHIFT;
    /// Bits 5 to 20: a swizzle mode, 0 for none. A retail title sets 1 to 4 for four of its chunk types (9, 11,
    /// 12 and 13 as zlib with swizzle 1, 3, 2 and 4). What the console does for them is not established.
    pub const SWIZZLE_SHIFT: u32 = 5;
    pub const SWIZZLE_MASK: u64 = 0xFFFF << SWIZZLE_SHIFT;
    /// The destination is an array of physical page frames, not an address.
    pub const DESTINATION_PAGES: u64 = 1 << 21;
    /// The source is memory, not a file.
    pub const MEMORY_SOURCE: u64 = 1 << 22;
    /// The memory source is an array of physical page frames.
    pub const SOURCE_PAGES: u64 = 1 << 23;
    /// Bits 24 and up: not declared.
    pub const RESERVED_MASK: u64 = !0 << 24;
}

/// One read of the project queue.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DecantRead {
    pub file: *mut c_void,
    pub offset: u64,
    pub source_size: u32,
    /// The caller retains this buffer until completion.
    pub destination: *mut c_void,
    pub destination_size: u32,
    pub codec: u32,
    pub bc_mode: u32,
}

const _: () = assert!(size_of::<DecantRead>() == 48);

pub type QueryInterfaceFn = unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT;
pub type AddRefFn = unsafe extern "system" fn(*mut c_void) -> u32;

#[repr(C)]
pub struct UnknownVtbl {
    pub query_interface: QueryInterfaceFn,
    pub add_ref: AddRefFn,
    pub release: AddRefFn,
}

/// `DecantFactory`: `OpenFile` 3, `CreateQueue` 4, `CreateStatusArray` 5.
#[repr(C)]
pub struct FactoryVtbl {
    pub base: UnknownVtbl,
    pub open_file: unsafe extern "system" fn(*mut c_void, *const u16, *mut *mut c_void) -> HRESULT,
    pub create_queue: unsafe extern "system" fn(*mut c_void, u16, *mut *mut c_void) -> HRESULT,
    pub create_status_array: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> HRESULT,
}

/// `DecantQueue`: `Enqueue` 3, `EnqueueStatus` 4, `EnqueueEvent` 5, `EnqueueFence` 6, `Submit` 7,
/// `EnqueueConsole` 8.
#[repr(C)]
pub struct QueueVtbl {
    pub base: UnknownVtbl,
    pub enqueue: unsafe extern "system" fn(*mut c_void, *const DecantRead) -> HRESULT,
    pub enqueue_status: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) -> HRESULT,
    pub enqueue_event: unsafe extern "system" fn(*mut c_void, HANDLE) -> HRESULT,
    pub enqueue_fence: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT,
    pub submit: unsafe extern "system" fn(*mut c_void) -> HRESULT,
    pub enqueue_console: unsafe extern "system" fn(*mut c_void, *const DecantConsoleRead) -> HRESULT,
}

/// `DecantStatus`: `Result` 3.
#[repr(C)]
pub struct StatusVtbl {
    pub base: UnknownVtbl,
    pub result: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT,
}

/// The vtable of a COM object, read from its first word.
///
/// # Safety
/// `object` must point to a COM object whose vtable is a `V`.
pub unsafe fn vtable<'a, V>(object: *mut c_void) -> &'a V {
    // SAFETY: the caller states the layout of the object and of its vtable.
    unsafe { &**(object as *const *const V) }
}

/// `Release` through the vtable of any COM object.
///
/// # Safety
/// `object` must be a live COM object, and the caller gives up one reference.
pub unsafe fn release(object: *mut c_void) -> u32 {
    // SAFETY: every COM vtable starts with the three `IUnknown` methods.
    unsafe { (vtable::<UnknownVtbl>(object).release)(object) }
}

/// A COM object of this DLL: a vtable pointer first, then its reference count.
pub trait Com: Sized {
    const IID: GUID;
    fn refs(&self) -> &AtomicU32;
    /// The address of the vtable of this type, to recognise its objects (`dynamic_cast`).
    fn vtable_address() -> *const c_void;
}

/// Whether `object` is a live looking object of type `T`: its vtable is the one of `T`.
///
/// # Safety
/// `object` must be null or point to a COM object (at least a vtable pointer).
pub unsafe fn is_object<T: Com>(object: *mut c_void) -> bool {
    // SAFETY: the caller states that a non null pointer is a COM object.
    !object.is_null() && unsafe { *(object as *const *const c_void) } == T::vtable_address()
}

/// `IUnknown::QueryInterface` of an object of type `T`: itself, or its own interface.
///
/// # Safety
/// Called through the vtable of a `T`: `this` is a `T`, `iid` readable, `out` null or writable.
pub unsafe extern "system" fn query_interface<T: Com>(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `out` is writable and `iid` readable, as the caller guarantees.
    unsafe {
        *out = std::ptr::null_mut();
        let wanted = guid_value(&*iid);
        if wanted != guid_value(&IID_IUNKNOWN) && wanted != guid_value(&T::IID) {
            crate::log::storage_log(&format!("QueryInterface iid={} hr=E_NOINTERFACE\n", guid_text(&*iid)));
            return E_NOINTERFACE;
        }
        add_ref::<T>(this);
        *out = this;
    }
    S_OK
}

/// # Safety
/// Called through the vtable of a `T`.
pub unsafe extern "system" fn add_ref<T: Com>(this: *mut c_void) -> u32 {
    // SAFETY: `this` is a live `T`.
    unsafe { (*(this as *const T)).refs().fetch_add(1, Ordering::Relaxed) + 1 }
}

/// # Safety
/// Called through the vtable of a `T` that was allocated by [`Ref::new`].
pub unsafe extern "system" fn release_object<T: Com>(this: *mut c_void) -> u32 {
    // SAFETY: `this` is a live `T`, and the last reference frees the box it was made in.
    unsafe {
        let n = (*(this as *const T)).refs().fetch_sub(1, Ordering::AcqRel) - 1;
        if n == 0 {
            drop(Box::from_raw(this as *mut T));
        }
        n
    }
}

/// An owned reference to an object of this DLL (the `ComPtr` of the C++).
pub struct Ref<T: Com>(NonNull<T>);

// SAFETY: the objects are shared between threads by design: their state sits behind locks or atomics.
unsafe impl<T: Com> Send for Ref<T> {}
// SAFETY: as above.
unsafe impl<T: Com> Sync for Ref<T> {}

impl<T: Com> Ref<T> {
    /// Takes ownership of a new object, whose count must be 1.
    pub fn new(value: T) -> Ref<T> {
        Ref(NonNull::from(Box::leak(Box::new(value))))
    }

    /// A new reference to the object at `raw`.
    ///
    /// # Safety
    /// `raw` must point to a live `T`.
    pub unsafe fn from_raw_add_ref(raw: *mut T) -> Ref<T> {
        // SAFETY: `raw` is a live `T`.
        unsafe { (*raw).refs().fetch_add(1, Ordering::Relaxed) };
        // SAFETY: the caller guarantees a non null pointer.
        Ref(unsafe { NonNull::new_unchecked(raw) })
    }

    /// Gives the reference to the caller as a raw pointer.
    pub fn into_raw(self) -> *mut T {
        let raw = self.0.as_ptr();
        std::mem::forget(self);
        raw
    }

    pub fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }
}

impl<T: Com> std::ops::Deref for Ref<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the reference keeps the object alive.
        unsafe { self.0.as_ref() }
    }
}

impl<T: Com> Clone for Ref<T> {
    fn clone(&self) -> Ref<T> {
        // SAFETY: the pointer is a live `T`, kept by `self`.
        unsafe { Ref::from_raw_add_ref(self.0.as_ptr()) }
    }
}

impl<T: Com> Drop for Ref<T> {
    fn drop(&mut self) {
        // SAFETY: gives back the reference this value owns.
        unsafe { release_object::<T>(self.0.as_ptr().cast()) };
    }
}

/// A Windows handle closed on drop.
pub struct Handle(pub HANDLE);

// SAFETY: a kernel handle may be used and closed from any thread.
unsafe impl Send for Handle {}
// SAFETY: as above.
unsafe impl Sync for Handle {}

impl Handle {
    pub const fn null() -> Handle {
        Handle(std::ptr::null_mut())
    }

    pub fn get(&self) -> HANDLE {
        self.0
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        let invalid = -1isize as HANDLE;
        if !self.0.is_null() && self.0 != invalid {
            // SAFETY: the handle is owned by this value and closed once.
            unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn win32_errors_become_hresults_as_the_macro_makes_them() {
        assert_eq!(hresult_from_win32(ERROR_HANDLE_EOF), 0x8007_0026_u32 as i32);
        assert_eq!(hresult_from_win32(ERROR_INVALID_DATA), 0x8007_000D_u32 as i32);
        assert_eq!(hresult_from_win32(ERROR_CANCELLED), 0x8007_04C7_u32 as i32);
        assert_eq!(hresult_from_win32(0), 0);
        assert_eq!(hresult_from_win32(0x8000_4005), 0x8000_4005_u32 as i32, "an HRESULT stays as it is");
    }

    #[test]
    fn guids_print_as_string_from_guid2_does() {
        assert_eq!(guid_text(&GUID::default()), "{00000000-0000-0000-0000-000000000000}");
        assert_eq!(guid_text(&IID_DECANT_FACTORY), "{BD498CD0-9410-4FDC-8800-B12BC02EB475}");
        assert_eq!(guid_text(&IID_IUNKNOWN), "{00000000-0000-0000-C000-000000000046}");
    }

    #[test]
    fn a_guid_value_is_the_number_its_text_spells() {
        let parsed = crate::spec::parse_guid("{bd498cd0-9410-4fdc-8800-b12bc02eb475}").expect("a guid");
        assert_eq!(parsed.0, guid_value(&IID_DECANT_FACTORY));
    }

    #[test]
    fn the_request_structures_keep_the_layout_of_the_c_header() {
        assert_eq!(size_of::<DecantConsoleRead>(), 72);
        assert_eq!(size_of::<DecantRead>(), 48);
        assert_eq!(std::mem::offset_of!(DecantRead, destination), 24);
        assert_eq!(std::mem::offset_of!(DecantRead, codec), 36);
    }
}
