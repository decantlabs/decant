//! The seven exports of `xgameruntime.dll` and the Windows calls behind them. Callers pass raw
//! pointers and the prototypes of most entry points are not public, so the code here is `unsafe`.
//! Each block states what it relies on. The ordinals come from `build.rs`.
#![allow(unsafe_code)]

use std::ffi::{OsStr, OsString, c_void};
use std::io::Write;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::Foundation::{FreeLibrary, GetLastError, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{DisableThreadLibraryCalls, GetModuleHandleW, GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY,
    PAGE_READWRITE, PAGE_WRITECOPY, VirtualQuery,
};
use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

use crate::tables::Guid;
use crate::{E_NOTIMPL, E_POINTER, NAMES, NO_RUNTIME_INIT, Query, REGDB_E_CLASSNOTREG, S_FALSE, S_OK, log_line, pass_line, query_lines};

/// `MAX_PATH` in UTF-16 units.
const MAX_PATH: usize = 260;

/// Indexes into `NAMES`, the C `enum`.
const X_INIT: usize = 0;
const X_INIT_EX: usize = 1;
const X_INIT_EX2: usize = 2;
const X_QUERY: usize = 3;
const X_UNINIT: usize = 4;
const X_ERRREPORT: usize = 5;
const X_UNLOAD: usize = 6;

/// A pass through entry: eight integer arguments and an integer result.
type Fn8 = unsafe extern "system" fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize;
type QueryFn = unsafe extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> i32;
type UnloadFn = unsafe extern "system" fn() -> i32;

/// The PC runtime's entry points as addresses, 0 when absent. Filled once, at the first call.
static TARGETS: OnceLock<[usize; NAMES.len()]> = OnceLock::new();
/// Serialises the writes to the log file (the SRWLOCK of the C code).
static LOG_LOCK: Mutex<()> = Mutex::new(());

/// This module's handle, from the linker defined image base.
fn this_module() -> HMODULE {
    unsafe extern "C" {
        static __ImageBase: u8;
    }
    // Only the address of the linker defined symbol is taken, it is never read.
    (&raw const __ImageBase) as HMODULE
}

/// What the C `DllMain` did on `DLL_PROCESS_ATTACH`. It runs from the C runtime start up code of
/// the DLL, which calls the `.CRT$XCU` initializers once when the process attaches. A `DllMain`
/// written in Rust would be exported by the `cdylib`, and the C DLL exports only the seven
/// entry points.
extern "C" fn on_process_attach() {
    // SAFETY: the handle is this module, loaded while this code runs.
    unsafe { DisableThreadLibraryCalls(this_module()) };
    crate::console_mode::install();
}

#[used]
#[unsafe(link_section = ".CRT$XCU")]
static ON_PROCESS_ATTACH: extern "C" fn() = on_process_attach;

/// The value of an environment variable when its length is 1 to `max` UTF-16 units, as the C
/// `GetEnvironmentVariableA` calls accepted it.
fn env_path(name: &str, max: usize) -> Option<OsString> {
    let value = std::env::var_os(name)?;
    (1..=max).contains(&value.encode_wide().count()).then_some(value)
}

/// Writes one log line to the debugger, then to `DECANT_XGR_LOG` when it names a file.
pub(crate) fn log(message: &str) {
    let line = log_line(message);
    decant_log::debug_string(&line);
    if let Some(path) = env_path("DECANT_XGR_LOG", MAX_PATH - 1) {
        let _guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut options = std::fs::OpenOptions::new();
        options.append(true).create(true);
        {
            use std::os::windows::fs::OpenOptionsExt;
            // FILE_SHARE_READ | FILE_SHARE_WRITE, so a reader may follow the log.
            options.share_mode(0x1 | 0x2);
        }
        // The C code wrote in text mode: every newline became CR LF.
        if let Ok(mut file) = options.open(path) {
            let _ = file.write_all(line.replace('\n', "\r\n").as_bytes());
        }
    }
}

/// Whether every region of `[p, p + n)` is committed and passes `ok`. The page protection of the
/// dirty tracking in `xmem.dll` can split one title range into several regions.
fn regions_ok(p: *const c_void, n: usize, mut ok: impl FnMut(&MEMORY_BASIC_INFORMATION) -> bool) -> bool {
    if p.is_null() || (p as usize) < 0x10000 {
        return false;
    }
    let Some(end) = (p as usize).checked_add(n.max(1)) else { return false };
    let mut at = p as usize;
    while at < end {
        // SAFETY: an all zero `MEMORY_BASIC_INFORMATION` is a valid value of plain integers.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `VirtualQuery` only inspects the address range and writes to `mbi`.
        if unsafe { VirtualQuery(at as *const c_void, &mut mbi, size_of::<MEMORY_BASIC_INFORMATION>()) } != size_of::<MEMORY_BASIC_INFORMATION>() {
            return false;
        }
        // A pool page of `xmem.dll` that the title never touched is only reserved in its view, and its
        // first access commits it: it counts as committed with the protection of the view. In this process
        // only `xmem.dll` leaves pages of mapped views reserved.
        if mbi.State == MEM_RESERVE && mbi.Type == MEM_MAPPED && mbi.AllocationProtect != 0 && mbi.AllocationProtect & PAGE_NOACCESS == 0 {
            mbi.State = MEM_COMMIT;
            mbi.Protect = mbi.AllocationProtect;
        }
        if mbi.State != MEM_COMMIT {
            return false;
        }
        let next = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize);
        if !ok(&mbi) || next <= at {
            return false;
        }
        at = next;
    }
    true
}

/// True when `[p, p + n)` is committed, readable memory.
pub(crate) fn readable(p: *const c_void, n: usize) -> bool {
    let readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    regions_ok(p, n, |mbi| mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && mbi.Protect & readable != 0)
}

/// Whether `xmem.dll` says `[p, p + n)` lies in memory the title may write, even where it made the
/// page read only for a while (a clean page of the GPU mirror). False without `xmem.dll`.
fn title_writable(p: *const c_void, n: usize) -> bool {
    // SAFETY: lookups of a module that is loaded or not, and a call of the export found, which only looks the range up.
    unsafe {
        let module = GetModuleHandleW(windows_sys::w!("xmem.dll"));
        if module.is_null() {
            return false;
        }
        GetProcAddress(module, c"DecantXmemGpuWritable".as_ptr().cast()).is_some_and(|f| {
            let f: unsafe extern "system" fn(usize, usize) -> i32 = std::mem::transmute(f);
            f(p as usize, n) != 0
        })
    }
}

/// Whether the `n` bytes at `p` are committed, writable memory.
fn writable(p: *const c_void, n: usize) -> bool {
    let readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let title = std::cell::OnceCell::new();
    regions_ok(p, n, |mbi| {
        mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0
            && mbi.Protect & readable != 0
            && (mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) != 0 || *title.get_or_init(|| title_writable(p, n)))
    })
}

/// A copy of the GUID at `p` when it is readable memory.
fn read_guid(p: *const Guid) -> Option<Guid> {
    // SAFETY: `readable` checked that the sixteen bytes are committed and readable. A GUID has no
    // invalid bit pattern, and the read does not assume the title aligned it.
    readable(p.cast(), size_of::<Guid>()).then(|| unsafe { p.read_unaligned() })
}

fn wide(text: &OsStr) -> Vec<u16> {
    text.encode_wide().chain(std::iter::once(0)).collect()
}

/// Loads the PC runtime and looks up its seven entry points. A missing runtime or entry point
/// is logged, and its slot stays 0.
fn load() -> [usize; NAMES.len()] {
    let mut targets = [0; NAMES.len()];
    let path = match env_path("DECANT_XGR_REAL", MAX_PATH * 2 - 1) {
        Some(path) => path,
        None => {
            let mut dir = [0u16; MAX_PATH];
            // SAFETY: `dir` has room for `MAX_PATH` units, which is what is passed.
            let n = unsafe { GetSystemDirectoryW(dir.as_mut_ptr(), MAX_PATH as u32) } as usize;
            if n == 0 || n >= MAX_PATH {
                log("no system directory; nothing to forward to");
                return targets;
            }
            let mut path = OsString::from_wide(&dir[..n]);
            path.push("\\xgameruntime.dll");
            path
        }
    };
    let shown = path.to_string_lossy().into_owned();
    // SAFETY: `wide` gives a zero terminated path that lives for the call.
    let module = unsafe { LoadLibraryExW(wide(&path).as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
    if module.is_null() {
        // SAFETY: no precondition.
        let error = unsafe { GetLastError() };
        log(&format!("PC runtime not loaded from {shown} (error {error}): initialisation accepted, only the local tables are served, every other table refused"));
        return targets;
    }
    if module == this_module() {
        log("DECANT_XGR_REAL points at this interposer; refusing to forward to itself");
        // SAFETY: the module was loaded just above and is released once.
        unsafe { FreeLibrary(module) };
        return targets;
    }
    for (slot, name) in targets.iter_mut().zip(NAMES) {
        let mut c_name = name.as_bytes().to_vec();
        c_name.push(0);
        // SAFETY: the module is loaded and `c_name` is a zero terminated name.
        *slot = unsafe { GetProcAddress(module, c_name.as_ptr()) }.map_or(0, |f| f as usize);
        if *slot == 0 {
            log(&format!("PC runtime has no {name}"));
        }
    }
    log(&format!("forwarding to {shown}"));
    targets
}

/// The PC runtime's address for entry `which`, 0 when there is none.
fn target(which: usize) -> usize {
    TARGETS.get_or_init(load)[which]
}

/// The `InitializeApiImpl*` / `UninitializeApiImpl` / `XErrorReport` prototypes are not public.
/// The x64 calling convention lets a pass through take eight integer arguments: the first four
/// are in registers, the others are read from the caller's stack and handed on unchanged, so a
/// call with fewer arguments loses nothing. (Floating point arguments would not survive, none is
/// expected on these entry points.)
#[allow(clippy::too_many_arguments)]
fn pass(which: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize, a6: usize, a7: usize, a8: usize) -> usize {
    let g1 = read_guid(a1 as *const Guid);
    let g2 = read_guid(a2 as *const Guid);
    let args = [a1, a2, a3, a4];
    let address = target(which);
    if address == 0 {
        // Without a PC runtime the initialisation is accepted, so the title goes on, and every
        // API table is then refused by `QueryApiImpl`: nothing is answered in its place.
        if matches!(which, X_INIT | X_INIT_EX | X_INIT_EX2) {
            log(&format!("{} {NO_RUNTIME_INIT}", pass_line(NAMES[which], args, g1.as_ref(), g2.as_ref(), Some(0))));
            return 0;
        }
        log(&pass_line(NAMES[which], args, g1.as_ref(), g2.as_ref(), None));
        decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("{} (no PC runtime)", NAMES[which]));
        return E_NOTIMPL as u32 as usize;
    }
    // SAFETY: the address is the PC runtime's export of this name, which takes at most the eight
    // integer arguments passed on, as the former C code assumed.
    let f = unsafe { std::mem::transmute::<usize, Fn8>(address) };
    // SAFETY: see above.
    let result = unsafe { f(a1, a2, a3, a4, a5, a6, a7, a8) };
    log(&pass_line(NAMES[which], args, g1.as_ref(), g2.as_ref(), Some(result)));
    result
}

macro_rules! pass_through {
    ($name:ident, $which:expr) => {
        /// # Safety
        /// Exported. The arguments are the title's, handed on to the PC runtime unchanged.
        #[allow(clippy::too_many_arguments)]
        #[unsafe(no_mangle)]
        pub unsafe extern "system" fn $name(a1: usize, a2: usize, a3: usize, a4: usize, a5: usize, a6: usize, a7: usize, a8: usize) -> usize {
            pass($which, a1, a2, a3, a4, a5, a6, a7, a8)
        }
    };
}

pass_through!(InitializeApiImpl, X_INIT);
pass_through!(InitializeApiImplEx, X_INIT_EX);
pass_through!(InitializeApiImplEx2, X_INIT_EX2);
pass_through!(UninitializeApiImpl, X_UNINIT);
pass_through!(XErrorReport, X_ERRREPORT);

/// `QueryApiImpl(api_id = RCX, riid = RDX, out = R8)`, as measured by ScarlettX on a console
/// title. The PC runtime exports the same name. The couple `(api_id, riid)` selects the table: the
/// log names the family from the `api_id` and says whether the riid is a listed version.
///
/// # Safety
/// Exported. `api_id` and `riid` are null or point to GUIDs, `out` is null or a writable pointer
/// location, as the title passes them to the PC runtime.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn QueryApiImpl(api_id: *const Guid, riid: *const Guid, out: *mut *mut c_void) -> i32 {
    // The loader may bind the title's imports after this DLL attached: patched again here.
    crate::console_mode::install();
    let api = read_guid(api_id);
    let riid_read = read_guid(riid);
    let address = target(X_QUERY);
    let mut local = None;
    // Offline play: the tables we have locally are served even beside a PC runtime, so the title
    // gets a signed in offline user instead of the Xbox sign in. They go together, as the local
    // XUser completes its calls through the local XAsync. DECANT_XGR_REAL_USER=1 forwards them.
    let offline = address != 0 && !real_user() && local_table_for(api.as_ref(), riid_read.as_ref());
    let result = if address == 0 || offline {
        let (result, served) = query_without_runtime(api.as_ref(), riid_read.as_ref(), out);
        local = served;
        result
    } else {
        // SAFETY: the address is the PC runtime's `QueryApiImpl`, which takes these arguments.
        let f = unsafe { std::mem::transmute::<usize, QueryFn>(address) };
        // SAFETY: the caller's arguments are passed on unchanged.
        let result = unsafe { f(api_id, riid, out) };
        if result >= 0 {
            // SAFETY: the PC runtime wrote its object to `out`.
            unsafe { crate::runtime_proxy::wrap_out(api.as_ref(), riid_read.as_ref(), out) };
        }
        result
    };
    let table = if readable(out.cast(), size_of::<*mut c_void>()) {
        // SAFETY: `readable` checked that `*out` is committed, readable memory.
        unsafe { *out as usize }
    } else {
        0
    };
    if result < 0 {
        decant_log::calls::record_with(decant_log::calls::Kind::Error, || {
            let g = |g: Option<&Guid>| g.map_or_else(|| "-".to_owned(), crate::guid_text);
            format!("QueryApiImpl(api {}, riid {}) -> {:#010x}", g(api.as_ref()), g(riid_read.as_ref()), result as u32)
        });
    }
    // The title asks for the same table some 70 times a second (XNetworking: 11,566 lines in 166 s,
    // a 60 MiB log): the first lines of each couple and every 1000th are written.
    let key = format!("{}/{}", api.as_ref().map_or_else(|| "-".to_owned(), crate::guid_text), riid_read.as_ref().map_or_else(|| "-".to_owned(), crate::guid_text));
    if query_line_due(&key) {
        for line in query_lines(&Query { api: api.as_ref(), riid: riid_read.as_ref(), result, table, forwarded: address != 0, local }) {
            log(&line);
        }
    }
    result
}

/// Lines written for each couple of API and interface at the start.
const QUERY_LINES_FIRST: u64 = 20;

/// Whether the call number `n` (from 1) of a couple is written to the log: the first
/// [`QUERY_LINES_FIRST`], then every 1000th (pure, for the tests).
fn query_line_number_due(n: u64) -> bool {
    n <= QUERY_LINES_FIRST || n.is_multiple_of(1000)
}

fn query_line_due(key: &str) -> bool {
    static COUNTS: std::sync::Mutex<Option<std::collections::HashMap<String, u64>>> = std::sync::Mutex::new(None);
    let mut g = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    let n = g.get_or_insert_with(Default::default).entry(key.to_owned()).or_insert(0);
    *n += 1;
    query_line_number_due(*n)
}

#[cfg(test)]
mod query_log_tests {
    #[test]
    fn a_repeated_query_is_written_at_first_and_then_every_thousandth_time() {
        let due: Vec<u64> = (1..=3000).filter(|&n| super::query_line_number_due(n)).collect();
        assert_eq!(due.len(), 20 + 3);
        assert_eq!((due[0], due[19], due[20], due[22]), (1, 20, 1000, 3000));
    }
}

/// The PC runtime's object for a couple, asked by us (not wrapped, not logged), null when none.
pub(crate) fn pc_query(api: &Guid, riid: &Guid) -> *mut c_void {
    let address = target(X_QUERY);
    let mut out = std::ptr::null_mut();
    if address != 0 {
        // SAFETY: the address is the PC runtime's `QueryApiImpl`; the GUIDs and `out` live for the call.
        let f = unsafe { std::mem::transmute::<usize, QueryFn>(address) };
        // SAFETY: see above.
        if unsafe { f(api, riid, &mut out) } < 0 {
            out = std::ptr::null_mut();
        }
    }
    out
}

/// `DECANT_XGR_REAL_USER=1`: the PC runtime answers XUser and the tables that go with it.
fn real_user() -> bool {
    static REAL: OnceLock<bool> = OnceLock::new();
    *REAL.get_or_init(|| std::env::var_os("DECANT_XGR_REAL_USER").is_some_and(|v| v == "1"))
}

/// Whether one of our local tables answers this couple.
fn local_table_for(api: Option<&Guid>, riid: Option<&Guid>) -> bool {
    crate::local_table::serves(api, riid)
        || crate::local_user::object_for(api, riid).is_some()
        || crate::local_networking::object_for(api, riid).is_some()
        || crate::local_display::object_for(api, riid).is_some()
        || crate::local_title::object_for(api, riid).is_some()
}

/// `QueryApiImpl` without a PC runtime: the XAsync / XTaskQueue / XThread couple
/// `({073B7DCB-...}, {073B7DCB-...})` gets the local table (`local_table`), the XUser couples
/// `({01ACD177-...}, {01ACD177-...})` and `({01ACD177-...}, {CEF4FAC0-...})` the local XUser
/// objects (`local_user`), the XNetworking couples `({37E56907-...}, {37E56907-...})` and
/// `({37E56907-...}, {BF2346B2-...})` the local offline XNetworking object (`local_networking`),
/// the XDisplay HDR couple `({03F0FE74-...}, {35F07670-...})` the local "HDR unavailable" object
/// (`local_display`),
/// every other table is refused as an unregistered class. `*out` is
/// written only when it is writable: the local object, or null for a refusal. The second value
/// names the local table served; serving one starts the heartbeat (`calllog`).
pub(crate) fn query_without_runtime(api: Option<&Guid>, riid: Option<&Guid>, out: *mut *mut c_void) -> (i32, Option<&'static str>) {
    let served = if crate::local_table::serves(api, riid) {
        Some((crate::local_table::object(), "XAsync/XTaskQueue/XThread"))
    } else {
        crate::local_user::object_for(api, riid)
            .or_else(|| crate::local_networking::object_for(api, riid))
            .or_else(|| crate::local_display::object_for(api, riid))
            .or_else(|| crate::local_title::object_for(api, riid))
    };
    if writable(out.cast(), size_of::<*mut c_void>()) {
        let object = served.map_or(std::ptr::null_mut(), |(object, _)| object);
        // SAFETY: `writable` checked that `*out` is committed, writable memory.
        unsafe { *out = object };
    } else if served.is_some() {
        // Nowhere to write the object: refused as a bad pointer.
        return (E_POINTER, None);
    }
    match served {
        Some((_, table)) => {
            crate::calllog::start_heartbeat();
            (S_OK, Some(table))
        }
        None => (REGDB_E_CLASSNOTREG, None),
    }
}

/// # Safety
/// Exported. No arguments.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DllCanUnloadNow() -> i32 {
    let address = target(X_UNLOAD);
    let result = if address == 0 {
        S_FALSE
    } else {
        // SAFETY: the address is the PC runtime's `DllCanUnloadNow`, which takes no argument.
        let f = unsafe { std::mem::transmute::<usize, UnloadFn>(address) };
        // SAFETY: see above.
        unsafe { f() }
    };
    log(&format!("DllCanUnloadNow -> 0x{:08x}", result as u32));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_table::XASYNC_TABLE;
    use crate::local_networking::XNETWORKING_BF2346B2;
    use crate::local_user::XUSER_GAMERTAG;

    const XUSER: Guid = Guid { data1: 0x01ACD177, data2: 0x91F9, data3: 0x4763, data4: [0xA3, 0x8E, 0xCC, 0xBB, 0x55, 0xCE, 0x32, 0xE0] };
    const XNETWORKING: Guid = Guid { data1: 0x37E56907, data2: 0x2F10, data3: 0x41E8, data4: [0xB7, 0x2F, 0x36, 0xED, 0xB1, 0x85, 0x33, 0x1A] };
    /// XStore and its context interface (`XStoreCreateContext`), from the PC GDK table.
    const XSTORE: Guid = Guid { data1: 0x0DD112AC, data2: 0x7C24, data3: 0x448C, data4: [0xB9, 0x2B, 0x39, 0x60, 0xFB, 0x5B, 0xD3, 0x0C] };
    const XSTORE_CONTEXT: Guid = Guid { data1: 0xB09D803C, data2: 0x2414, data3: 0x4A05, data4: [0x82, 0xC6, 0x66, 0xDF, 0xDC, 0x9E, 0x9A, 0x44] };
    /// The XUser interface of `XUserIsStoreUser`: not served.
    const XUSER_STORE: Guid = Guid { data1: 0x079415E3, data2: 0x6727, data3: 0x437F, data4: [0x8E, 0x9D, 0x8F, 0x8F, 0x9B, 0x24, 0x39, 0xF7] };

    #[test]
    fn without_a_pc_runtime_the_xasync_and_xuser_tables_are_served() {
        let mut out = std::ptr::dangling_mut::<c_void>();
        assert_eq!(query_without_runtime(Some(&XASYNC_TABLE), Some(&XASYNC_TABLE), &mut out), (S_OK, Some("XAsync/XTaskQueue/XThread")));
        assert_eq!(out, crate::local_table::object());
        // The served object answers its slots, `this` first.
        // SAFETY: `out` is the local object, its first field the vtable of 32 slots.
        let get_status = unsafe { *(*(out as *const *const usize)).add(3) };
        assert_ne!(get_status, 0);

        out = std::ptr::dangling_mut::<c_void>();
        assert_eq!(query_without_runtime(Some(&XUSER), Some(&XUSER), &mut out), (S_OK, Some("XUser")));
        assert_eq!(out, crate::local_user::object());
        out = std::ptr::dangling_mut::<c_void>();
        assert_eq!(query_without_runtime(Some(&XUSER), Some(&XUSER_GAMERTAG), &mut out), (S_OK, Some("XUser gamertag")));
        assert_eq!(out, crate::local_user::gamertag_object());
    }

    #[test]
    fn without_a_pc_runtime_both_xnetworking_couples_get_the_offline_object() {
        for riid in [XNETWORKING, XNETWORKING_BF2346B2] {
            let mut out = std::ptr::dangling_mut::<c_void>();
            assert_eq!(query_without_runtime(Some(&XNETWORKING), Some(&riid), &mut out), (S_OK, Some("XNetworking")));
            assert_eq!(out, crate::local_networking::object());
        }
        assert_eq!(query_without_runtime(Some(&XNETWORKING), Some(&XNETWORKING), std::ptr::null_mut()), (E_POINTER, None));
    }

    #[test]
    fn without_a_pc_runtime_the_store_context_interface_gets_the_local_store() {
        let mut out = std::ptr::dangling_mut::<c_void>();
        assert_eq!(query_without_runtime(Some(&XSTORE), Some(&XSTORE_CONTEXT), &mut out), (S_OK, Some("XStore")));
        assert!(!out.is_null());
        assert!(local_table_for(Some(&XSTORE), Some(&XSTORE_CONTEXT)), "served beside a PC runtime too");
    }

    #[test]
    fn without_a_pc_runtime_other_couples_are_refused() {
        let unknown = Guid { data1: 0x12345678, data2: 0x9ABC, data3: 0xDEF0, data4: [1, 2, 3, 4, 5, 6, 7, 8] };
        let mut out = std::ptr::dangling_mut::<c_void>();
        for (api, riid) in [(XNETWORKING, XSTORE), (XNETWORKING_BF2346B2, XNETWORKING_BF2346B2), (unknown, unknown), (XSTORE, XSTORE), (XUSER, XUSER_STORE), (XASYNC_TABLE, XUSER), (XUSER, XASYNC_TABLE)] {
            out = std::ptr::dangling_mut::<c_void>();
            assert_eq!(query_without_runtime(Some(&api), Some(&riid), &mut out), (REGDB_E_CLASSNOTREG, None));
            assert!(out.is_null(), "a refused table leaves a null object");
        }
        assert_eq!(query_without_runtime(Some(&XASYNC_TABLE), None, &mut out), (REGDB_E_CLASSNOTREG, None));
        assert_eq!(query_without_runtime(Some(&XUSER), None, &mut out), (REGDB_E_CLASSNOTREG, None));
        assert_eq!(query_without_runtime(Some(&XNETWORKING), None, &mut out), (REGDB_E_CLASSNOTREG, None));
        assert_eq!(query_without_runtime(Some(&XASYNC_TABLE), Some(&XASYNC_TABLE), std::ptr::null_mut()), (E_POINTER, None));
        assert_eq!(query_without_runtime(Some(&XUSER), Some(&XUSER), std::ptr::null_mut()), (E_POINTER, None));
    }
}
