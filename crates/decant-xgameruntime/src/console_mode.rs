//! The console model a title reads from the registry. Code linked into the title (not a runtime
//! call) reads the DWORD `ConsoleMode` below `HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM`
//! through `RegOpenKeyW`, `RegQueryValueExW` and `RegCloseKey` of `ADVAPI32.dll` and maps it to a
//! device type (0x64 is an Xbox Series X). A PC has no such value, so the title would not know its
//! device. This module redirects those three import entries of the title executable to functions
//! that present the key and the value like a Series X does, and forward every other key unchanged.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};

type RegKey = *mut c_void;
type OpenKeyFn = unsafe extern "system" fn(RegKey, *const u16, *mut RegKey) -> u32;
type QueryValueFn = unsafe extern "system" fn(RegKey, *const u16, *mut u32, *mut u32, *mut u8, *mut u32) -> u32;
type CloseKeyFn = unsafe extern "system" fn(RegKey) -> u32;

/// The predefined handle `HKEY_LOCAL_MACHINE`.
const HKLM: usize = 0xffff_ffff_8000_0002;
/// Registry path the title opens.
pub const KEY: &str = r"Software\Microsoft\Windows NT\CurrentVersion\OEM";
/// Name of the value the title reads.
pub const VALUE: &str = "ConsoleMode";
/// The device type of an Xbox Series X.
pub const SERIES_X: u32 = 0x64;

const ERROR_SUCCESS: u32 = 0;
const ERROR_FILE_NOT_FOUND: u32 = 2;
const ERROR_INVALID_PARAMETER: u32 = 87;
const ERROR_MORE_DATA: u32 = 234;
const REG_DWORD: u32 = 4;

/// Names of the redirected imports. The position is the slot in [`FORWARD`].
const IMPORTS: [&str; 3] = ["RegOpenKeyW", "RegQueryValueExW", "RegCloseKey"];
/// The original function behind each redirected import, 0 while the import is not redirected.
static FORWARD: [AtomicUsize; 3] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];
/// Whether the first read of the value was already logged.
static LOGGED: AtomicBool = AtomicBool::new(false);

/// The handle given out for our key. Its address cannot collide with a real registry handle.
static KEY_MARKER: u8 = 0;

fn our_handle() -> RegKey {
    (&raw const KEY_MARKER).cast_mut().cast()
}

/// Whether the import descriptor of `dll` can carry the registry functions.
pub fn carries_registry(dll: &str) -> bool {
    let name = dll.to_ascii_lowercase();
    name == "advapi32.dll" || (name.starts_with("api-ms-win-core-registry-") && name.ends_with(".dll"))
}

/// Reads a zero terminated UTF-16 string. A null pointer gives `None`.
///
/// # Safety
/// `ptr` is null or points to a string with a terminating zero unit.
unsafe fn read_utf16(ptr: *const u16) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let mut units = Vec::new();
    let mut at = ptr;
    // SAFETY: the caller guarantees a terminator, so every unit up to it is readable.
    unsafe {
        while *at != 0 {
            units.push(*at);
            at = at.add(1);
        }
    }
    Some(String::from_utf16_lossy(&units))
}

/// Whether a root handle and a sub key path address the key we answer for.
pub fn matches_key(root: usize, path: Option<&str>) -> bool {
    root == HKLM && path.is_some_and(|p| p.eq_ignore_ascii_case(KEY))
}

/// What a value read on our key returns.
///
/// `offered` is the buffer size the caller passes (`None` when it passes no size), `has_buffer`
/// tells whether a data buffer was passed. The result is the status and whether to store the DWORD.
pub fn value_reply(name: Option<&str>, offered: Option<u32>, has_buffer: bool) -> (u32, bool) {
    if !name.is_some_and(|n| n.eq_ignore_ascii_case(VALUE)) {
        return (ERROR_FILE_NOT_FOUND, false);
    }
    match (offered, has_buffer) {
        (None, true) => (ERROR_INVALID_PARAMETER, false),
        (Some(size), true) if size < 4 => (ERROR_MORE_DATA, false),
        (_, store) => (ERROR_SUCCESS, store),
    }
}

/// Calls the original function of slot `slot` after casting it to `F`.
///
/// # Safety
/// The slot holds a function of type `F`, which is so once the import is redirected.
unsafe fn original<F: Copy>(slot: usize) -> F {
    // SAFETY: a function pointer has the size of usize, and the slot was stored by `redirect`.
    unsafe { std::mem::transmute_copy::<usize, F>(&FORWARD[slot].load(Ordering::Acquire)) }
}

unsafe extern "system" fn open_key(root: RegKey, path: *const u16, out: *mut RegKey) -> u32 {
    // SAFETY: the title passes a terminated path or null.
    let text = unsafe { read_utf16(path) };
    if matches_key(root as usize, text.as_deref()) && !out.is_null() {
        // SAFETY: not null, and the title offers room for one handle.
        unsafe { out.write(our_handle()) };
        return ERROR_SUCCESS;
    }
    // SAFETY: this function is only installed after slot 0 was filled.
    unsafe { original::<OpenKeyFn>(0)(root, path, out) }
}

unsafe extern "system" fn query_value(key: RegKey, name: *const u16, reserved: *mut u32, kind: *mut u32, data: *mut u8, size: *mut u32) -> u32 {
    if key != our_handle() {
        // SAFETY: this function is only installed after slot 1 was filled.
        return unsafe { original::<QueryValueFn>(1)(key, name, reserved, kind, data, size) };
    }
    // SAFETY: the title passes valid or null pointers and sizes `data` by `size`.
    unsafe {
        let offered = if size.is_null() { None } else { Some(size.read()) };
        let (status, store) = value_reply(read_utf16(name).as_deref(), offered, !data.is_null());
        if matches!(status, ERROR_SUCCESS | ERROR_MORE_DATA) {
            if !kind.is_null() {
                kind.write(REG_DWORD);
            }
            if !size.is_null() {
                size.write(4);
            }
        }
        if store {
            data.cast::<u32>().write_unaligned(SERIES_X);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                crate::exports::log(&format!("the title read {VALUE}, reported {SERIES_X:#x} (Xbox Series X)"));
            }
        }
        status
    }
}

unsafe extern "system" fn close_key(key: RegKey) -> u32 {
    if key == our_handle() {
        return ERROR_SUCCESS;
    }
    // SAFETY: this function is only installed after slot 2 was filled.
    unsafe { original::<CloseKeyFn>(2)(key) }
}

/// The replacement function for import slot `slot`.
fn replacement(slot: usize) -> usize {
    match slot {
        0 => open_key as *const () as usize,
        1 => query_value as *const () as usize,
        _ => close_key as *const () as usize,
    }
}

/// Reads a zero terminated ASCII name of at most 255 bytes.
///
/// # Safety
/// The bytes at `ptr` are readable up to the terminator or 256 bytes.
unsafe fn read_ascii(ptr: *const u8) -> Option<String> {
    let mut bytes = Vec::new();
    for i in 0..256 {
        // SAFETY: by the contract above.
        let byte = unsafe { *ptr.add(i) };
        if byte == 0 {
            return String::from_utf8(bytes).ok();
        }
        bytes.push(byte);
    }
    None
}

/// Writes `new` into the pointer slot, restoring the page protection afterwards.
///
/// # Safety
/// `slot` points to an 8 byte import address entry of the loaded image.
unsafe fn swap_slot(slot: *mut usize, new: usize) -> Option<usize> {
    let mut previous_protection = 0u32;
    // SAFETY: the entry lies in a mapped image.
    if unsafe { VirtualProtect(slot.cast(), 8, PAGE_READWRITE, &mut previous_protection) } == 0 {
        return None;
    }
    // SAFETY: the page is writable now.
    let old = unsafe { slot.read_unaligned() };
    unsafe {
        slot.write_unaligned(new);
        VirtualProtect(slot.cast(), 8, previous_protection, &mut previous_protection);
    }
    Some(old)
}

/// Redirects the matching entries of one import descriptor and returns how many were changed.
///
/// # Safety
/// `base` is the image base, and `lookup_rva` and `address_rva` are the name table and the address
/// table of one import descriptor of that image.
unsafe fn redirect_descriptor(base: usize, lookup_rva: u32, address_rva: u32) -> usize {
    let mut changed = 0;
    let mut index = 0usize;
    loop {
        // SAFETY: both tables end with a zero entry and are read up to it.
        let entry = unsafe { ((base + lookup_rva as usize + index * 8) as *const u64).read_unaligned() };
        if entry == 0 {
            return changed;
        }
        // An entry with the top bit set imports by ordinal and has no name.
        if entry >> 63 == 0 {
            // SAFETY: the hint/name record starts with a 2 byte hint, then the name.
            let name = unsafe { read_ascii((base + entry as u32 as usize + 2) as *const u8) };
            if let Some(slot_no) = name.and_then(|n| IMPORTS.iter().position(|&i| i == n)) {
                let slot = (base + address_rva as usize + index * 8) as *mut usize;
                // SAFETY: the address table entry belongs to the same import.
                let current = unsafe { slot.read_unaligned() };
                // Before binding the entry still holds the RVA of the name, a small number.
                if current != replacement(slot_no) && current > u32::MAX as usize {
                    // SAFETY: the entry is an import address of the loaded image.
                    if unsafe { swap_slot(slot, replacement(slot_no)) }.is_some() {
                        FORWARD[slot_no].store(current, Ordering::Release);
                        changed += 1;
                    }
                }
            }
        }
        index += 1;
    }
}

/// Redirects the three registry imports of the title executable. Calling it again is harmless
/// because an entry that already points to its replacement is skipped.
pub fn install() {
    // SAFETY: a null name selects the executable of this process.
    let base = unsafe { GetModuleHandleW(std::ptr::null()) } as usize;
    if base == 0 {
        return;
    }
    let mut changed = 0;
    // SAFETY: the headers and the import directory of a loaded image are readable, and every read
    // stays inside the structures they describe.
    unsafe {
        if (base as *const u16).read_unaligned() != 0x5A4D {
            return;
        }
        let nt = base + ((base + 0x3C) as *const u32).read_unaligned() as usize;
        let is_pe64 = (nt as *const u32).read_unaligned() == 0x0000_4550 && ((nt + 0x18) as *const u16).read_unaligned() == 0x20B;
        if !is_pe64 {
            return;
        }
        let directory = ((nt + 0x90) as *const u32).read_unaligned();
        if directory == 0 {
            return;
        }
        let mut descriptor = base + directory as usize;
        loop {
            let lookup = (descriptor as *const u32).read_unaligned();
            let dll_name = ((descriptor + 12) as *const u32).read_unaligned();
            let address = ((descriptor + 16) as *const u32).read_unaligned();
            if dll_name == 0 {
                break;
            }
            if lookup != 0 && read_ascii((base + dll_name as usize) as *const u8).is_some_and(|d| carries_registry(&d)) {
                changed += redirect_descriptor(base, lookup, address);
            }
            descriptor += 20;
        }
    }
    if changed > 0 {
        crate::exports::log(&format!("{changed} registry imports now present {KEY}\\{VALUE} as an Xbox Series X"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf16z(text: &str) -> Vec<u16> {
        text.encode_utf16().chain([0]).collect()
    }

    #[test]
    fn key_match_needs_hklm_and_the_oem_path() {
        assert!(matches_key(HKLM, Some(KEY)));
        assert!(matches_key(HKLM, Some(&KEY.to_ascii_uppercase())));
        assert!(!matches_key(HKLM, Some(r"Software\Microsoft\Windows NT")));
        assert!(!matches_key(HKLM, None));
        assert!(!matches_key(0xffff_ffff_8000_0001, Some(KEY)));
    }

    #[test]
    fn only_registry_dlls_are_searched() {
        assert!(carries_registry("Advapi32.DLL"));
        assert!(carries_registry("API-MS-WIN-CORE-REGISTRY-L1-1-0.DLL"));
        assert!(!carries_registry("api-ms-win-core-file-l1-1-0.dll"));
        assert!(!carries_registry("user32.dll"));
    }

    #[test]
    fn value_reply_covers_each_buffer_situation() {
        assert_eq!(value_reply(Some("CONSOLEMODE"), Some(8), true), (ERROR_SUCCESS, true));
        assert_eq!(value_reply(Some(VALUE), Some(3), true), (ERROR_MORE_DATA, false));
        assert_eq!(value_reply(Some(VALUE), None, true), (ERROR_INVALID_PARAMETER, false));
        assert_eq!(value_reply(Some(VALUE), None, false), (ERROR_SUCCESS, false));
        assert_eq!(value_reply(Some(VALUE), Some(1), false), (ERROR_SUCCESS, false));
        assert_eq!(value_reply(Some("Model"), Some(4), true), (ERROR_FILE_NOT_FOUND, false));
        assert_eq!(value_reply(None, Some(4), true), (ERROR_FILE_NOT_FOUND, false));
    }

    #[test]
    fn open_query_close_roundtrip_on_our_key() {
        let path = utf16z(KEY);
        let name = utf16z(VALUE);
        let mut handle: RegKey = std::ptr::null_mut();
        let (mut kind, mut data, mut size) = (0u32, 0u32, 4u32);
        // SAFETY: valid pointers, and our key never reaches a forwarded function.
        unsafe {
            assert_eq!(open_key(HKLM as RegKey, path.as_ptr(), &mut handle), ERROR_SUCCESS);
            assert_eq!(handle, our_handle());
            let status = query_value(handle, name.as_ptr(), std::ptr::null_mut(), &mut kind, (&raw mut data).cast(), &mut size);
            assert_eq!(status, ERROR_SUCCESS);
            assert_eq!(close_key(handle), ERROR_SUCCESS);
        }
        assert_eq!((kind, data, size), (REG_DWORD, SERIES_X, 4));
    }

    #[test]
    fn a_short_buffer_reports_the_needed_size() {
        let name = utf16z(VALUE);
        let (mut kind, mut size) = (0u32, 2u32);
        let mut data = [0u8; 2];
        // SAFETY: valid pointers, the handle is ours.
        let status = unsafe { query_value(our_handle(), name.as_ptr(), std::ptr::null_mut(), &mut kind, data.as_mut_ptr(), &mut size) };
        assert_eq!((status, kind, size, data), (ERROR_MORE_DATA, REG_DWORD, 4, [0, 0]));
    }
}
