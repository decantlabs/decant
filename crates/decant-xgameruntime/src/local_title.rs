//! What the local XPackage, XGame, XSystem, Accessibility and XStore tables share: the title as it runs
//! here (its id from `MicrosoftGame.config`, its folder), the COM style object those tables hand
//! out, the slot that answers a named "not implemented", and the string and pointer helpers of the
//! GDK getters.
//!
//! With Gaming Services installed the title, a plain unpackaged exe, would get these tables from
//! the PC runtime, which has no package identity for it. [`object_for`] is the one entry `exports.rs` asks.
//!
//! Our own code.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::E_NOTIMPL;
use crate::calllog::Counters;
use crate::exports::log;
use crate::tables::Guid;
use crate::taskqueue::{E_INVALIDARG, S_OK};

/// `E_NOT_SUFFICIENT_BUFFER`.
pub(crate) const E_NOT_SUFFICIENT_BUFFER: i32 = 0x8007_007A_u32 as i32;
const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;
const E_POINTER: i32 = 0x8000_4003_u32 as i32;

/// The most slots a local table of this group has, padding included.
pub(crate) const MAX_SLOTS: usize = 96;

/// The object for a couple one of the local tables of this group serves, with the table's name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    crate::local_package::object_for(api, riid)
        .or_else(|| crate::local_game::object_for(api, riid))
        .or_else(|| crate::local_system::object_for(api, riid))
        .or_else(|| crate::local_accessibility::object_for(api, riid))
        .or_else(|| crate::local_gamesave::object_for(api, riid))
        .or_else(|| crate::local_pls::object_for(api, riid))
        .or_else(|| crate::local_store::object_for(api, riid))
}

// The title.

/// The title as it runs here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Title {
    /// `<TitleId>` of `MicrosoftGame.config`, 0 when it has none.
    pub(crate) id: u32,
    /// The folder of the executable: the package root, as `gdk-launch` stages it.
    pub(crate) folder: PathBuf,
    /// `<StoreId>` of `MicrosoftGame.config`, empty when it has none.
    pub(crate) store_id: String,
}

/// The title of this process: `MicrosoftGame.config` next to the executable, read once.
pub(crate) fn title() -> &'static Title {
    static TITLE: OnceLock<Title> = OnceLock::new();
    TITLE.get_or_init(|| {
        let folder = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)).unwrap_or_default();
        let title = title_in(&folder);
        log(&format!("title: id {:08X} (package identifier {}), folder {}", title.id, package_identifier_of(title.id), title.folder.display()));
        title
    })
}

/// The title whose `MicrosoftGame.config` lies in `folder`. A missing config or title id is
/// logged and gives id 0.
pub(crate) fn title_in(folder: &Path) -> Title {
    let config = folder.join("MicrosoftGame.config");
    let (id, store_id) = match decant_gdk_config::read_gdk_config(&config) {
        Ok(c) => {
            let id = parse_title_id(&c.title_id).unwrap_or_else(|| {
                log(&format!("{}: no usable TitleId ({:?}), title id 0", config.display(), c.title_id));
                0
            });
            (id, c.store_id.trim().to_owned())
        }
        Err(e) => {
            log(&format!("title id 0: {e}"));
            (0, String::new())
        }
    };
    Title { id, folder: folder.to_path_buf(), store_id }
}

/// A `<TitleId>` value: hexadecimal, with or without `0x`.
pub(crate) fn parse_title_id(text: &str) -> Option<u32> {
    let t = text.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    u32::from_str_radix(t, 16).ok()
}

/// The package identifier of a title: its id as eight upper case hex digits.
pub(crate) fn package_identifier_of(id: u32) -> String {
    format!("{id:08X}")
}

/// The package identifier of this title.
pub(crate) fn package_identifier() -> String {
    package_identifier_of(title().id)
}

// The object and its slots.

/// A vtable as the thunks read it: one function address per slot.
#[repr(transparent)]
pub(crate) struct Methods<const N: usize>(pub(crate) [*const (); N]);

// SAFETY: the addresses of functions, never written.
unsafe impl<const N: usize> Sync for Methods<N> {}

/// The COM style object: a pointer to its vtable.
#[repr(C)]
pub(crate) struct Object {
    vtbl: *const c_void,
}

// SAFETY: the object only holds the address of a static vtable.
unsafe impl Sync for Object {}

impl Object {
    pub(crate) const fn new<const N: usize>(vtbl: &'static Methods<N>) -> Self {
        Object { vtbl: (vtbl as *const Methods<N>).cast() }
    }

    /// The address written to `*out`.
    pub(crate) fn as_out(&'static self) -> *mut c_void {
        (self as *const Object).cast_mut().cast()
    }
}

/// A table of this group, as its generic slots see it.
pub(crate) trait LocalTable {
    /// The name in the log.
    const NAME: &'static str;
    /// The interface ids `QueryInterface` answers with the object.
    const IIDS: &'static [Guid];
    /// The call counters of the slots.
    fn counts() -> &'static Counters<MAX_SLOTS>;
    /// The GDK function of `slot`, `None` when it is not known.
    fn slot_name(slot: usize) -> Option<&'static str>;
    /// The object.
    fn object() -> *mut c_void;
}

/// Slot 0: the interface ids of the table answer the object.
pub(crate) unsafe extern "system" fn query_interface<T: LocalTable>(_this: *const c_void, iid: *const Guid, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: COM contract: `iid` is null or a GUID, `out` a writable pointer location.
    unsafe {
        let iid = if iid.is_null() { None } else { Some(iid.read_unaligned()) };
        if iid.is_some_and(|i| T::IIDS.contains(&i)) {
            *out = T::object();
            S_OK
        } else {
            *out = std::ptr::null_mut();
            E_NOINTERFACE
        }
    }
}

/// Slots 1 and 2: AddRef and Release of a static object.
pub(crate) unsafe extern "system" fn no_count(_this: *const c_void) -> u32 {
    1
}

/// A slot not served: logs the table, the slot and its GDK name when known (bounded, `calllog`)
/// and answers `E_NOTIMPL`. Never a crash.
pub(crate) unsafe extern "system" fn not_implemented<T: LocalTable, const SLOT: usize>(_this: *const c_void) -> i32 {
    if let Some(n) = T::counts().next(SLOT) {
        log(&not_implemented_line(T::NAME, SLOT, T::slot_name(SLOT), n));
    }
    E_NOTIMPL
}

/// The line of a slot not served.
pub(crate) fn not_implemented_line(table: &str, slot: usize, name: Option<&str>, n: u64) -> String {
    format!("local {table} table: slot {slot} {} is not implemented -> E_NOTIMPL [call {n}]", name.unwrap_or("(unknown)"))
}

/// The address of a slot function, for a [`Methods`] entry.
macro_rules! slot {
    ($f:expr) => {
        $f as *const ()
    };
}
pub(crate) use slot;

// Helpers of the GDK getters.

/// Writes `value` through `out` when it is not null; `E_INVALIDARG` for null.
pub(crate) fn put<T>(out: *mut T, value: T) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: not null; the title passes room for a `T`, aligned or not.
    unsafe { out.write_unaligned(value) };
    S_OK
}

/// Copies `text` and its NUL into the title's `size` byte buffer and writes the bytes it takes to
/// `used` (when not null), as the runtime's string getters do. Too small a buffer gives
/// `E_NOT_SUFFICIENT_BUFFER` and `used` still names the size needed.
pub(crate) fn put_string(text: &str, size: usize, buffer: *mut u8, used: *mut usize) -> i32 {
    let len = text.len() + 1;
    if !used.is_null() {
        // SAFETY: not null, the title's location.
        unsafe { used.write_unaligned(len) };
    }
    if buffer.is_null() || size < len {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds `size >= len` bytes.
    unsafe {
        buffer.copy_from_nonoverlapping(text.as_ptr(), text.len());
        buffer.add(text.len()).write(0);
    }
    S_OK
}

/// The title's NUL terminated string, `None` for null.
pub(crate) fn text(s: *const c_char) -> Option<String> {
    // SAFETY: the title passes a NUL terminated string or null.
    (!s.is_null()).then(|| unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned())
}

/// A registration token for a callback nobody raises here.
pub(crate) fn token() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Calls slot `slot` of `object` as the title's thunk does. Tests only.
#[cfg(test)]
pub(crate) fn method<F: Copy>(object: *mut c_void, slot: usize) -> F {
    assert_eq!(size_of::<F>(), size_of::<usize>());
    // SAFETY: the object's first field is its vtable, which covers the slots the tests call; the
    // caller names the slot's signature.
    unsafe { std::mem::transmute_copy(&*(*(object as *const *const *const ())).add(slot)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_title_id_is_read_from_the_config_next_to_the_exe() {
        let dir = std::env::temp_dir().join(format!("decant_xgr_title_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = r#"<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <Identity Name="Contoso.SyntheticTitle" Publisher="CN=Contoso" Version="1.0.0.0"/>
  <ExecutableList>
    <Executable Name="Synthetic.exe" TargetDeviceFamily="Scarlett"/>
  </ExecutableList>
  <TitleId>0ABC1234</TitleId>
  <StoreId>9NSYNTHETIC0</StoreId>
</Game>
"#;
        std::fs::write(dir.join("MicrosoftGame.config"), config).unwrap();
        assert_eq!(title_in(&dir), Title { id: 0x0ABC_1234, folder: dir.clone(), store_id: "9NSYNTHETIC0".into() });
        std::fs::remove_file(dir.join("MicrosoftGame.config")).unwrap();
        assert_eq!(title_in(&dir), Title { id: 0, folder: dir.clone(), store_id: String::new() }, "no config: id 0, no store id");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn title_ids_are_hex_with_or_without_a_prefix() {
        assert_eq!(parse_title_id(" 0x7F43C2B1 "), Some(0x7F43_C2B1));
        assert_eq!(parse_title_id("7f43c2b1"), Some(0x7F43_C2B1));
        assert_eq!(parse_title_id(""), None);
        assert_eq!(parse_title_id("XYZ"), None);
        assert_eq!(package_identifier_of(0xABC), "00000ABC");
    }

    #[test]
    fn strings_need_room_for_their_nul() {
        let mut buffer = [0xFFu8; 8];
        let mut used = 0;
        assert_eq!(put_string("RETAIL", 6, buffer.as_mut_ptr(), &mut used), E_NOT_SUFFICIENT_BUFFER);
        assert_eq!(used, 7);
        assert_eq!(put_string("RETAIL", 7, buffer.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(&buffer[..7], b"RETAIL\0");
        assert_eq!(put_string("x", 8, std::ptr::null_mut(), std::ptr::null_mut()), E_NOT_SUFFICIENT_BUFFER);
        assert_eq!(put(std::ptr::null_mut::<u32>(), 1), E_INVALIDARG);
    }

    #[test]
    fn a_slot_not_implemented_is_named() {
        assert_eq!(
            not_implemented_line("XPackage", 13, Some("XPackageEnumerateChunkAvailability"), 1),
            "local XPackage table: slot 13 XPackageEnumerateChunkAvailability is not implemented -> E_NOTIMPL [call 1]"
        );
        assert_eq!(not_implemented_line("XGame", 7, None, 2), "local XGame table: slot 7 (unknown) is not implemented -> E_NOTIMPL [call 2]");
    }
}
