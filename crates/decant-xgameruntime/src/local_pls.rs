//! Offline XPersistentLocalStorage table: `api_id = {F4FAF4D4-2D04-4FCE-B3E0-474A713A3E84}`,
//! interfaces `{41A4E10C-5A7E-41D9-8C37-37BDE62A07D6}` and `{D29411DF-0794-4553-8B27-95FC02D0F75D}`.
//! The PC runtime refuses persistent local storage for an unpackaged process, so this table serves
//! it from a folder on the host.
//!
//! * The folder is `DECANT_PLS_DIR` when the launcher set it, otherwise
//!   `%LOCALAPPDATA%\Decant\<title id>\pls` (`decant_gdk_config::persistent_local_storage_folder`).
//!   It is created on first use.
//! * The path handed to the title is the root of the drive that `decant-launch` maps over the
//!   folder for the run (`DECANT_PLS_ROOT`, for example `T:\`, like `G:` over the game folder).
//!   Without a mapping it is the host folder with a trailing backslash. A buffer that is too small
//!   gives `E_FAIL`, the documented console answer.
//! * The quota is `<PersistentLocalStorage><SizeMB>` of `MicrosoftGame.config`, 4 GiB when absent.
//!   Used space is the size of the files in the folder, total free is the quota minus that, and
//!   available free is the total free capped by what the host disk has left.
//! * Prompting for space completes at once on our XAsync with `S_OK` when the quota has room for
//!   the requested bytes and `E_ABORT` otherwise, since nobody can be asked. The result slot reads
//!   the block.
//!
//! Slots 3 to 7 follow the order of the public GDK `XPersistentLocalStorage.h`, which agrees with
//! the offsets 0x20 (path) and 0x28 (space) seen in title calls. Slots 8 to 11 log a named "not
//! implemented". Every call is logged with a bound (`calllog`).
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::local_title::{LocalTable, MAX_SLOTS, Methods, Object, no_count, not_implemented, put, query_interface, slot, title};
use crate::tables::Guid;
use crate::taskqueue::{E_ABORT, E_FAIL, S_OK};
use crate::xasync::{self as xa, XAsyncBlock};

/// The table's `api_id`.
pub(crate) const PLS_TABLE: Guid = Guid { data1: 0xF4FAF4D4, data2: 0x2D04, data3: 0x4FCE, data4: [0xB3, 0xE0, 0x47, 0x4A, 0x71, 0x3A, 0x3E, 0x84] };
/// The current interface of the table.
pub(crate) const PLS_IID: Guid = Guid { data1: 0x41A4E10C, data2: 0x5A7E, data3: 0x41D9, data4: [0x8C, 0x37, 0x37, 0xBD, 0xE6, 0x2A, 0x07, 0xD6] };
/// The interface other titles ask for, with the same slots.
pub(crate) const PLS_IID_OLDER: Guid = Guid { data1: 0xD29411DF, data2: 0x0794, data3: 0x4553, data4: [0x8B, 0x27, 0x95, 0xFC, 0x02, 0xD0, 0xF7, 0x5D] };

/// The host folder, set by `decant-launch`.
pub(crate) const DIR_ENV: &str = "DECANT_PLS_DIR";
/// The drive root over the host folder, set by `decant-launch` when it mapped one.
pub(crate) const ROOT_ENV: &str = "DECANT_PLS_ROOT";
/// The quota without `<SizeMB>` in the config.
const DEFAULT_QUOTA_MB: u64 = 4096;

const NAME: &str = "XPersistentLocalStorage";
const SLOTS: usize = 12;
static COUNTS: Counters<MAX_SLOTS> = Counters::new();
static OBJECT: Object = Object::new(&VTBL);

/// The object for the couples this table serves, with its name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    (api == Some(&PLS_TABLE) && riid.is_some_and(|r| *r == PLS_IID || *r == PLS_IID_OLDER)).then(|| (OBJECT.as_out(), NAME))
}

struct Pls;

impl LocalTable for Pls {
    const NAME: &'static str = NAME;
    const IIDS: &'static [Guid] = &[PLS_IID, PLS_IID_OLDER];
    fn counts() -> &'static Counters<MAX_SLOTS> {
        &COUNTS
    }
    fn slot_name(slot: usize) -> Option<&'static str> {
        SLOT_NAMES.iter().find(|(number, _)| *number == slot).map(|&(_, name)| name)
    }
    fn object() -> *mut c_void {
        OBJECT.as_out()
    }
}
/// GDK function names by slot, only for the slots that are identified.
const SLOT_NAMES: [(usize, &str); 8] = [
    (0, "QueryInterface"),
    (1, "AddRef"),
    (2, "Release"),
    (3, "XPersistentLocalStorageGetPathSize"),
    (4, "XPersistentLocalStorageGetPath"),
    (5, "XPersistentLocalStorageGetSpaceInfo"),
    (6, "XPersistentLocalStoragePromptUserForSpaceAsync"),
    (7, "XPersistentLocalStoragePromptUserForSpaceResult"),
];

/// The storage as this process serves it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Storage {
    /// The host folder.
    pub(crate) folder: PathBuf,
    /// The path the title gets, ending in a backslash.
    pub(crate) path: String,
    /// The quota in bytes.
    pub(crate) quota: u64,
}

/// The storage for a host folder, a drive root (when one is mapped and reachable) and a quota in
/// MiB. The folder is made.
pub(crate) fn storage_in(folder: PathBuf, root: Option<String>, quota_mb: u64) -> Storage {
    if let Err(e) = std::fs::create_dir_all(&folder) {
        log(&format!("local {NAME} table: folder {} could not be made ({e})", folder.display()));
    }
    let root = root.map(|r| r.trim().to_string()).filter(|r| !r.is_empty() && Path::new(r).is_dir());
    let mut path = root.unwrap_or_else(|| folder.to_string_lossy().into_owned());
    if !path.ends_with('\\') {
        path.push('\\');
    }
    Storage { folder, path, quota: quota_mb.saturating_mul(1 << 20) }
}

/// The storage of this title, read once.
fn serving_storage() -> &'static Storage {
    static STORAGE: OnceLock<Storage> = OnceLock::new();
    STORAGE.get_or_init(|| {
        let title = title();
        let env = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        let fallback = || std::env::temp_dir().join("Decant").join(format!("{:08X}", title.id)).join("pls");
        // The tests never write to the user's data folder.
        let folder = if cfg!(test) { fallback() } else { env(DIR_ENV).map(PathBuf::from).or_else(|| decant_gdk_config::persistent_local_storage_folder(title.id)).unwrap_or_else(fallback) };
        let quota_mb = decant_gdk_config::read_gdk_config(&title.folder.join("MicrosoftGame.config")).ok().and_then(|c| c.persistent_local_storage_mb).unwrap_or(DEFAULT_QUOTA_MB);
        let s = storage_in(folder, env(ROOT_ENV), quota_mb);
        log(&format!("local {NAME} table: path {} over {}, quota {} MiB", s.path, s.folder.display(), quota_mb));
        if !s.path.is_ascii() {
            log(&format!("local {NAME} table: the path is not ASCII; the title reads it in its code page"));
        }
        s
    })
}

/// `XPersistentLocalStorageSpaceInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpaceInfo {
    /// Writable now: the quota's rest, as far as the host disk allows.
    pub(crate) available_free_bytes: u64,
    /// The quota's rest.
    pub(crate) total_free_bytes: u64,
    pub(crate) used_bytes: u64,
    /// The quota.
    pub(crate) total_bytes: u64,
}

/// Total size of the files below `root`, at most eight levels deep.
fn folder_usage(root: &Path) -> u64 {
    let mut pending = vec![(root.to_path_buf(), 0u32)];
    let mut total = 0;
    while let Some((dir, depth)) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_file() {
                total += meta.len();
            } else if meta.is_dir() && depth < 8 {
                pending.push((entry.path(), depth + 1));
            }
        }
    }
    total
}

// What the host disk under `dir` has free for this user, unlimited when unknown.
fn host_disk_free(dir: &Path) -> u64 {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
    let mut free = 0u64;
    // SAFETY: a NUL terminated path and a writable output; the other outputs may be null.
    let ok = unsafe { windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) };
    if ok != 0 { free } else { u64::MAX }
}

/// The space of `s` now.
pub(crate) fn space_of(s: &Storage) -> SpaceInfo {
    let used = folder_usage(&s.folder);
    let rest = s.quota.saturating_sub(used);
    SpaceInfo { available_free_bytes: rest.min(host_disk_free(&s.folder)), total_free_bytes: rest, used_bytes: used, total_bytes: s.quota }
}

type This = *const c_void;
type Block = *mut XAsyncBlock;

/// Tells our prompts apart from other calls on the same block.
static PROMPT_ID: u8 = 0;

unsafe fn report_path_size(out: *mut usize) -> i32 {
    put(out, serving_storage().path.len() + 1)
}

/// Too small a buffer gives `E_FAIL` (the console's answer), `used` still names the size needed.
unsafe fn copy_path(size: usize, buffer: *mut u8, used: *mut usize) -> i32 {
    let path = &serving_storage().path;
    let len = path.len() + 1;
    if !used.is_null() {
        // SAFETY: not null, the title's location.
        unsafe { used.write_unaligned(len) };
    }
    if buffer.is_null() || size < len {
        return E_FAIL;
    }
    // SAFETY: the buffer holds `size >= len` bytes.
    unsafe {
        buffer.copy_from_nonoverlapping(path.as_ptr(), path.len());
        buffer.add(path.len()).write(0);
    }
    S_OK
}

unsafe fn report_space(out: *mut SpaceInfo) -> i32 {
    put(out, space_of(serving_storage()))
}

/// Granted when the quota has room for `bytes`, else declined: there is nobody to ask.
unsafe fn ask_for_space(bytes: u64, block: Block) -> i32 {
    let room = space_of(serving_storage()).total_free_bytes;
    let result = if bytes <= room { Ok(Vec::new()) } else { Err(E_ABORT) };
    // SAFETY: the title's block, valid until the call completes.
    unsafe { xa::begin_with(block, &raw const PROMPT_ID as usize, move || result) }
}

unsafe fn space_prompt_result(block: Block) -> i32 {
    // SAFETY: the title's block; no payload to copy.
    unsafe { xa::XAsyncGetResult(block, (&raw const PROMPT_ID).cast(), 0, std::ptr::null_mut(), std::ptr::null_mut()) }
}

logged_shim!(NAME, COUNTS, This, 3 as "XPersistentLocalStorageGetPathSize", path_size_entry => report_path_size, (out: *mut usize) -> i32);
logged_shim!(NAME, COUNTS, This, 4 as "XPersistentLocalStorageGetPath", path_entry => copy_path, (size: usize, buffer: *mut u8, used: *mut usize) -> i32);
logged_shim!(NAME, COUNTS, This, 5 as "XPersistentLocalStorageGetSpaceInfo", space_info_entry => report_space, (out: *mut SpaceInfo) -> i32);
logged_shim!(NAME, COUNTS, This, 6 as "XPersistentLocalStoragePromptUserForSpaceAsync", prompt_entry => ask_for_space, (bytes: u64, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 7 as "XPersistentLocalStoragePromptUserForSpaceResult", prompt_result_entry => space_prompt_result, (block: Block) -> i32);

static VTBL: Methods<SLOTS> = Methods([
    slot!(query_interface::<Pls>),
    slot!(no_count),
    slot!(no_count),
    slot!(path_size_entry),
    slot!(path_entry),
    slot!(space_info_entry),
    slot!(prompt_entry),
    slot!(prompt_result_entry),
    slot!(not_implemented::<Pls, 8>),
    slot!(not_implemented::<Pls, 9>),
    slot!(not_implemented::<Pls, 10>),
    slot!(not_implemented::<Pls, 11>),
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exports::query_without_runtime;
    use crate::local_title::method;
    use crate::xasync::XAsyncGetStatus;

    fn table(riid: &Guid) -> *mut c_void {
        let mut out = std::ptr::null_mut();
        assert_eq!(query_without_runtime(Some(&PLS_TABLE), Some(riid), &mut out), (S_OK, Some(NAME)));
        assert!(!out.is_null());
        out
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("decant_xgr_pls_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn both_interfaces_are_served_and_nothing_else() {
        assert_eq!(table(&PLS_IID), table(&PLS_IID_OLDER));
        assert!(object_for(Some(&PLS_TABLE), Some(&PLS_TABLE)).is_none());
        assert!(object_for(Some(&PLS_IID), Some(&PLS_IID)).is_none());
        assert_eq!(size_of::<SpaceInfo>(), 32);
    }

    #[test]
    fn the_folder_is_made_and_shown_as_the_drive_root_when_there_is_one() {
        let folder = temp("root");
        let s = storage_in(folder.clone(), None, 1);
        assert!(folder.is_dir());
        assert_eq!(s.path, format!("{}\\", folder.display()));
        assert_eq!(s.quota, 1 << 20);
        // A root that is reachable is what the title gets (here the folder itself, as a mapped drive would show it).
        let root = format!("{}\\", folder.display());
        assert_eq!(storage_in(folder.clone(), Some(root.clone()), 1).path, root);
        // A root that is not there (no mapping) falls back to the host folder.
        assert_eq!(storage_in(folder.clone(), Some("Q:\\decant_xgr_no_such_drive\\".into()), 1).path, format!("{}\\", folder.display()));
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn space_counts_the_files_against_the_quota() {
        let folder = temp("space");
        let s = storage_in(folder.clone(), None, 1);
        std::fs::write(folder.join("a.dat"), [0u8; 1000]).unwrap();
        std::fs::create_dir_all(folder.join("sub")).unwrap();
        std::fs::write(folder.join("sub").join("b.dat"), [0u8; 24]).unwrap();
        let info = space_of(&s);
        assert_eq!((info.used_bytes, info.total_bytes, info.total_free_bytes), (1024, 1 << 20, (1 << 20) - 1024));
        assert!(info.available_free_bytes <= info.total_free_bytes);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn the_slots_answer_path_space_and_prompts() {
        type Size = unsafe extern "system" fn(This, *mut usize) -> i32;
        type Get = unsafe extern "system" fn(This, usize, *mut u8, *mut usize) -> i32;
        type Space = unsafe extern "system" fn(This, *mut SpaceInfo) -> i32;
        type Prompt = unsafe extern "system" fn(This, u64, Block) -> i32;
        type Result = unsafe extern "system" fn(This, Block) -> i32;
        let t = table(&PLS_IID);
        // SAFETY: each slot gets the arguments of its GDK function.
        unsafe {
            let mut size = 0usize;
            assert_eq!(method::<Size>(t, 3)(t, &mut size), S_OK);
            assert_eq!(size, serving_storage().path.len() + 1);
            let mut buffer = vec![0u8; size];
            let mut used = 0usize;
            assert_eq!(method::<Get>(t, 4)(t, size - 1, buffer.as_mut_ptr(), &mut used), E_FAIL);
            assert_eq!(used, size);
            assert_eq!(method::<Get>(t, 4)(t, size, buffer.as_mut_ptr(), &mut used), S_OK);
            assert_eq!(&buffer[..size - 1], serving_storage().path.as_bytes());
            assert_eq!(buffer[size - 1], 0);
            assert!(serving_storage().path.ends_with('\\'));

            let mut info = SpaceInfo::default();
            assert_eq!(method::<Space>(t, 5)(t, &mut info), S_OK);
            assert_eq!(info.total_bytes, serving_storage().quota);
            assert_eq!(method::<Space>(t, 5)(t, std::ptr::null_mut()), crate::taskqueue::E_INVALIDARG);

            for (bytes, expect) in [(1, S_OK), (u64::MAX, E_ABORT)] {
                let mut block = XAsyncBlock { queue: std::ptr::null_mut(), context: std::ptr::null_mut(), callback: 0, internal: [0; 32] };
                assert_eq!(method::<Prompt>(t, 6)(t, bytes, &mut block), S_OK);
                assert_eq!(XAsyncGetStatus(&mut block, true), expect);
                assert_eq!(method::<Result>(t, 7)(t, &mut block), expect, "{bytes} bytes");
            }
        }
    }
}
