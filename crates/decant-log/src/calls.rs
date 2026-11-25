//! `DECANT_CALLS=1`: a per run count of every shim answer that did not really answer, across all the
//! Decant DLLs of a process. A stub path calls [`not_implemented`], [`stub`] or [`error`] with what
//! was asked (an export name, or an interface and slot). Each distinct question gets one entry with
//! its call count, the time of its first call and its first caller outside the shim (`module+rva`).
//!
//! The entries live in one named shared memory block per process (`Local\decant-calls-<pid>`), so
//! the DLLs (each with its own copy of this crate) count into the same table, and the launcher, which
//! opens the same block, can still read it after the process ended. The first DLL that records
//! anything starts a thread that writes the report to `decant_calls.txt` once a second while the table
//! changes and every 30 seconds otherwise. `decant-launch` writes it once more at the exit.
//!
//! Off by default. Off, a stub path pays one relaxed atomic load.

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

/// The environment switch.
pub const ENV: &str = "DECANT_CALLS";
/// A file to write the report to instead of `decant_calls.txt`.
pub const PATH_ENV: &str = "DECANT_CALLS_PATH";
/// The report's file name, in the folder of `DECANT_LOG_PATH` or else the current folder.
pub const FILE_NAME: &str = "decant_calls.txt";

/// Entries one process can hold. Further distinct questions are only counted as overflow.
pub const SLOTS: usize = 2048;
const MAGIC: u32 = 0x5358_4331;
const MODULE_WORDS: usize = 4;
const TEXT_WORDS: usize = 12;

/// What a stub path answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum Kind {
    /// `E_NOTIMPL`, `ERROR_CALL_NOT_IMPLEMENTED` or a slot nobody wrote.
    NotImplemented = 1,
    /// A made up value: a fixed answer, zero, an empty result.
    Stub = 2,
    /// A real error code returned for a question the shim could not serve.
    Error = 3,
}

impl Kind {
    fn from_u32(v: u32) -> Option<Kind> {
        match v {
            1 => Some(Kind::NotImplemented),
            2 => Some(Kind::Stub),
            3 => Some(Kind::Error),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Kind::NotImplemented => "not implemented",
            Kind::Stub => "stub value",
            Kind::Error => "error",
        }
    }
}

/// Whether `value` of [`ENV`] switches the counter on: set, not empty and not `0`, `off`, `false`, `no`.
pub fn switch_on(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        let v = v.trim().to_ascii_lowercase();
        !v.is_empty() && !matches!(v.as_str(), "0" | "off" | "false" | "no")
    })
}

static STATE: AtomicU8 = AtomicU8::new(0);

/// Whether [`ENV`] is on in this process (read once).
#[inline]
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = switch_on(std::env::var(ENV).ok().as_deref());
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Counts a question the shim answered with "not implemented".
#[inline]
pub fn not_implemented(what: &str) {
    if enabled() {
        record_slow(Kind::NotImplemented, what);
    }
}

/// Counts a question the shim answered with a made up value.
#[inline]
pub fn stub(what: &str) {
    if enabled() {
        record_slow(Kind::Stub, what);
    }
}

/// Counts a question the shim answered with an error.
#[inline]
pub fn error(what: &str) {
    if enabled() {
        record_slow(Kind::Error, what);
    }
}

/// Like the three above, with the text built only when the counter is on.
#[inline]
pub fn record_with(kind: Kind, what: impl FnOnce() -> String) {
    if enabled() {
        record_slow(kind, &what());
    }
}

/// One entry of the shared table: 256 bytes, all atomics, so zeroed memory is an empty slot.
#[repr(C)]
pub struct Slot {
    /// 0 empty, 1 being written, 2 ready.
    state: AtomicU32,
    kind: AtomicU32,
    hash: AtomicU64,
    count: AtomicU64,
    /// Milliseconds since the Unix epoch of the first call.
    first_ms: AtomicU64,
    module: [AtomicU64; MODULE_WORDS],
    name: [AtomicU64; TEXT_WORDS],
    caller: [AtomicU64; TEXT_WORDS],
}

/// The shared table.
#[repr(C)]
pub struct Table {
    magic: AtomicU32,
    /// Set by the DLL that runs the report thread.
    writer: AtomicU32,
    entries: AtomicU32,
    _reserved: AtomicU32,
    overflow: AtomicU64,
    _pad: [AtomicU64; 5],
    slots: [Slot; SLOTS],
}

/// Bytes of the shared block.
pub const TABLE_BYTES: usize = std::mem::size_of::<Table>();

fn hash(module: &str, name: &str, kind: Kind) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in module.bytes().chain([0]).chain(name.bytes()).chain([0, kind as u8]) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h | 1
}

fn pack(words: &[AtomicU64], text: &str) {
    let mut bytes = vec![0u8; words.len() * 8];
    // Cut at a character boundary, keep one zero at the end.
    let mut end = text.len().min(bytes.len() - 1);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    bytes[..end].copy_from_slice(&text.as_bytes()[..end]);
    for (w, chunk) in words.iter().zip(bytes.as_chunks::<8>().0) {
        w.store(u64::from_le_bytes(*chunk), Ordering::Relaxed);
    }
}

fn unpack(words: &[AtomicU64]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.load(Ordering::Relaxed).to_le_bytes()).collect();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn same(words: &[AtomicU64], text: &str) -> bool {
    let room = words.len() * 8 - 1;
    let mut end = text.len().min(room);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    unpack(words) == text[..end]
}

/// One entry as read back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub kind: Kind,
    pub module: String,
    pub name: String,
    pub count: u64,
    pub first_ms: u64,
    pub caller: String,
}

impl Table {
    fn check_magic(&self) {
        let _ = self.magic.compare_exchange(0, MAGIC, Ordering::AcqRel, Ordering::Relaxed);
    }

    /// The ready slot for this question, if it is in the table.
    fn find(&self, h: u64, module: &str, name: &str, kind: Kind) -> Result<&Slot, usize> {
        let start = (h as usize) % SLOTS;
        for i in 0..SLOTS {
            let index = (start + i) % SLOTS;
            let slot = &self.slots[index];
            let mut state = slot.state.load(Ordering::Acquire);
            let mut spins = 0u32;
            while state == 1 && spins < 1 << 20 {
                std::hint::spin_loop();
                spins += 1;
                state = slot.state.load(Ordering::Acquire);
            }
            match state {
                0 => return Err(index),
                2 if slot.hash.load(Ordering::Relaxed) == h
                    && slot.kind.load(Ordering::Relaxed) == kind as u32
                    && same(&slot.module, module)
                    && same(&slot.name, name) =>
                {
                    return Ok(slot);
                }
                _ => {}
            }
        }
        Err(SLOTS)
    }

    /// Counts one call. `caller` is asked for only when the question is new. Returns true when it was.
    pub fn record(&self, kind: Kind, module: &str, name: &str, now_ms: u64, caller: impl FnOnce() -> String) -> bool {
        self.check_magic();
        let h = hash(module, name, kind);
        let mut from = match self.find(h, module, name, kind) {
            Ok(slot) => {
                slot.count.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            Err(SLOTS) => {
                self.overflow.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            Err(index) => index,
        };
        let caller = caller();
        loop {
            let slot = &self.slots[from];
            if slot.state.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                slot.hash.store(h, Ordering::Relaxed);
                slot.kind.store(kind as u32, Ordering::Relaxed);
                slot.first_ms.store(now_ms, Ordering::Relaxed);
                pack(&slot.module, module);
                pack(&slot.name, name);
                pack(&slot.caller, &caller);
                slot.count.store(1, Ordering::Relaxed);
                slot.state.store(2, Ordering::Release);
                self.entries.fetch_add(1, Ordering::AcqRel);
                return true;
            }
            // Another thread took this slot first: it may have been the same question.
            match self.find(h, module, name, kind) {
                Ok(slot) => {
                    slot.count.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                Err(SLOTS) => {
                    self.overflow.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                Err(index) => from = index,
            }
        }
    }

    /// The number of distinct questions so far.
    pub fn entries(&self) -> u32 {
        self.entries.load(Ordering::Acquire)
    }

    /// Every ready entry, the earliest first call first.
    pub fn snapshot(&self) -> Vec<Entry> {
        let mut out: Vec<Entry> = self
            .slots
            .iter()
            .filter(|s| s.state.load(Ordering::Acquire) == 2)
            .filter_map(|s| {
                Some(Entry {
                    kind: Kind::from_u32(s.kind.load(Ordering::Relaxed))?,
                    module: unpack(&s.module),
                    name: unpack(&s.name),
                    count: s.count.load(Ordering::Relaxed),
                    first_ms: s.first_ms.load(Ordering::Relaxed),
                    caller: unpack(&s.caller),
                })
            })
            .collect();
        out.sort_by(|a, b| a.first_ms.cmp(&b.first_ms).then_with(|| a.module.cmp(&b.module)).then_with(|| a.name.cmp(&b.name)));
        out
    }

    /// The report text, with times relative to `start_ms` (the process start, Unix milliseconds).
    pub fn report(&self, start_ms: u64, heading: &str) -> String {
        format_report(&self.snapshot(), self.overflow.load(Ordering::Relaxed), start_ms, heading)
    }
}

/// The report: one line per entry in the order of the first call, then the ten most called.
pub fn format_report(entries: &[Entry], overflow: u64, start_ms: u64, heading: &str) -> String {
    let mut out = format!("{heading}: {} distinct unanswered questions", entries.len());
    if overflow > 0 {
        out += &format!(", {overflow} calls not counted (table full)");
    }
    out += "\nfirst call  count     answer           shim and question  <-  first caller\n";
    for e in entries {
        let at = e.first_ms.saturating_sub(start_ms) as f64 / 1000.0;
        out += &format!("{at:>9.3}s {:>7}  {:<16} {}!{}  <-  {}\n", e.count, e.kind.label(), e.module, e.name, e.caller);
    }
    let mut busy: Vec<&Entry> = entries.iter().collect();
    busy.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.first_ms.cmp(&b.first_ms)));
    if busy.len() > 1 {
        out += "most called:\n";
        for e in busy.into_iter().take(10) {
            out += &format!("  {:>9}  {}!{} ({})\n", e.count, e.module, e.name, e.kind.label());
        }
    }
    out
}

/// The name of the shared block of process `pid`.
pub fn mapping_name(pid: u32) -> String {
    format!("Local\\decant-calls-{pid}")
}

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// Where the report goes: [`PATH_ENV`], else [`FILE_NAME`] next to `DECANT_LOG_PATH`, else in
/// the current folder.
pub fn report_path(path: Option<std::ffi::OsString>, log_path: Option<std::ffi::OsString>) -> std::path::PathBuf {
    if let Some(p) = path.filter(|p| !p.is_empty()) {
        return p.into();
    }
    match log_path.filter(|p| !p.is_empty()).map(std::path::PathBuf::from).and_then(|p| p.parent().map(std::path::Path::to_path_buf)) {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(FILE_NAME),
        _ => FILE_NAME.into(),
    }
}

#[cfg(windows)]
pub use platform::{Shared, caller_outside_this_module, module_of, process_start_ms};

#[inline(never)]
#[cold]
fn record_slow(kind: Kind, what: &str) {
    #[cfg(windows)]
    platform::record(kind, what);
    #[cfg(not(windows))]
    let _ = (kind, what);
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod platform {
    use super::*;
    use std::ffi::c_void;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, HMODULE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW, GetModuleHandleExW};
    use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentProcessId, GetProcessTimes};

    /// The shared block, mapped. Dropping it unmaps the view and closes the handle.
    pub struct Shared {
        handle: HANDLE,
        view: *mut c_void,
    }

    // SAFETY: the view is a plain shared memory block accessed through atomics only.
    unsafe impl Send for Shared {}
    // SAFETY: as above.
    unsafe impl Sync for Shared {}

    impl Shared {
        /// Creates or opens the block of process `pid`.
        pub fn open(pid: u32) -> Option<Shared> {
            let name: Vec<u16> = mapping_name(pid).encode_utf16().chain([0]).collect();
            // SAFETY: a pagefile backed section with a NUL terminated name.
            let handle = unsafe { CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, TABLE_BYTES as u32, name.as_ptr()) };
            if handle.is_null() {
                return None;
            }
            // SAFETY: maps the whole section just created or opened.
            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, TABLE_BYTES) }.Value;
            if view.is_null() {
                // SAFETY: our handle.
                unsafe { CloseHandle(handle) };
                return None;
            }
            Some(Shared { handle, view })
        }

        pub fn table(&self) -> &Table {
            // SAFETY: the view is TABLE_BYTES long, page aligned, zero filled when new, and every
            // field of Table is an atomic, valid for any bit pattern.
            unsafe { &*self.view.cast::<Table>() }
        }
    }

    impl Drop for Shared {
        fn drop(&mut self) {
            // SAFETY: the view and handle are ours.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.view });
                CloseHandle(self.handle);
            }
        }
    }

    fn shared() -> Option<&'static Shared> {
        static SHARED: OnceLock<Option<Shared>> = OnceLock::new();
        // SAFETY: plain call.
        SHARED.get_or_init(|| Shared::open(unsafe { GetCurrentProcessId() })).as_ref()
    }

    /// The module holding `address` and its base, `None` outside any module.
    pub fn module_of(address: usize) -> Option<(HMODULE, String)> {
        let mut module: HMODULE = std::ptr::null_mut();
        // SAFETY: FROM_ADDRESS takes the address as the name argument; the reference count is kept.
        if unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, address as *const u16, &mut module) } == 0 {
            return None;
        }
        let mut buffer = [0u16; 520];
        // SAFETY: the buffer length is passed.
        let n = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) } as usize;
        let path = String::from_utf16_lossy(&buffer[..n.min(buffer.len())]);
        let name = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_owned();
        Some((module, name))
    }

    /// The first return address on this thread's stack outside the module that holds this crate's
    /// copy, as `module+rva`: who called into the shim.
    #[inline(never)]
    pub fn caller_outside_this_module() -> String {
        let here = caller_outside_this_module as *const () as usize;
        let own = module_of(here).map(|(m, _)| m);
        let mut frames = [std::ptr::null_mut::<c_void>(); 48];
        // SAFETY: the buffer holds the number of frames asked for.
        let n = unsafe { RtlCaptureStackBackTrace(1, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) } as usize;
        for &frame in &frames[..n] {
            let at = frame as usize;
            match module_of(at) {
                Some((m, name)) if Some(m) != own => return format!("{name}+{:#x}", at - m as usize),
                Some(_) => {}
                None => return format!("{at:#x}"),
            }
        }
        "?".into()
    }

    /// This process's start, Unix milliseconds.
    pub fn process_start_ms() -> u64 {
        let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut created, mut a, mut b, mut c) = (zero, zero, zero, zero);
        // SAFETY: four FILETIME out parameters.
        if unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut a, &mut b, &mut c) } == 0 {
            return 0;
        }
        let ticks = (u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime);
        (ticks / 10_000).saturating_sub(11_644_473_600_000)
    }

    fn own_module_name() -> String {
        module_of(own_module_name as *const () as usize).map_or_else(|| "?".into(), |(_, n)| n)
    }

    pub(super) fn record(kind: Kind, what: &str) {
        let Some(shared) = shared() else { return };
        static MODULE: OnceLock<String> = OnceLock::new();
        let module = MODULE.get_or_init(own_module_name);
        let table = shared.table();
        table.record(kind, module, what, now_ms(), caller_outside_this_module);
        if table.writer.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
            let _ = std::thread::Builder::new().name("decant calls report".into()).spawn(move || writer(shared));
        }
    }

    /// Writes the report once a second while the table changes, and every 30 seconds.
    fn writer(shared: &'static Shared) {
        let path = report_path(std::env::var_os(PATH_ENV), std::env::var_os("DECANT_LOG_PATH"));
        let start = process_start_ms();
        // SAFETY: plain call.
        let pid = unsafe { GetCurrentProcessId() };
        let mut written = u32::MAX;
        let mut last = Instant::now();
        loop {
            let entries = shared.table().entries();
            if entries != written || last.elapsed() >= Duration::from_secs(30) {
                let text = shared.table().report(start, &format!("DECANT_CALLS process {pid} at {:.1} s", now_ms().saturating_sub(start) as f64 / 1000.0));
                let _ = std::fs::write(&path, text);
                written = entries;
                last = Instant::now();
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Box<Table> {
        // SAFETY-free: every field is an atomic, and zero is a valid value for each.
        let words = TABLE_BYTES / 8;
        let v: Vec<AtomicU64> = (0..words).map(|_| AtomicU64::new(0)).collect();
        let boxed: Box<[AtomicU64]> = v.into_boxed_slice();
        from_words(boxed)
    }

    #[allow(unsafe_code)]
    fn from_words(words: Box<[AtomicU64]>) -> Box<Table> {
        assert_eq!(words.len() * 8, TABLE_BYTES);
        // SAFETY: same size, alignment 8 suits Table (no field needs more), all atomics, zeroed.
        unsafe { Box::from_raw(Box::into_raw(words).cast::<Table>()) }
    }

    #[test]
    fn the_layout_is_fixed() {
        assert_eq!(std::mem::size_of::<Slot>(), 256);
        assert_eq!(TABLE_BYTES, 64 + 256 * SLOTS);
    }

    #[test]
    fn the_switch_is_off_unless_set_to_something() {
        assert!(switch_on(Some("1")));
        assert!(switch_on(Some("yes")));
        for off in [None, Some(""), Some("0"), Some("OFF"), Some(" no ")] {
            assert!(!switch_on(off), "{off:?}");
        }
    }

    #[test]
    fn one_entry_per_question_with_its_first_caller() {
        let t = table();
        let mut asked = 0;
        assert!(t.record(Kind::NotImplemented, "d3d12_xs.dll", "device slot 55", 1000, || {
            asked += 1;
            "title.exe+0x10".into()
        }));
        for _ in 0..4 {
            assert!(!t.record(Kind::NotImplemented, "d3d12_xs.dll", "device slot 55", 2000, || "other".into()));
        }
        assert!(t.record(Kind::Stub, "d3d12_xs.dll", "device slot 55", 3000, || "title.exe+0x20".into()));
        assert!(t.record(Kind::Error, "xmem.dll", "XMemTransferMemory", 500, || "title.exe+0x30".into()));
        assert_eq!(asked, 1);
        assert_eq!(t.entries(), 3);
        let s = t.snapshot();
        assert_eq!(s[0].name, "XMemTransferMemory");
        assert_eq!(s[1], Entry { kind: Kind::NotImplemented, module: "d3d12_xs.dll".into(), name: "device slot 55".into(), count: 5, first_ms: 1000, caller: "title.exe+0x10".into() });
        assert_eq!(s[2].kind, Kind::Stub);
        let r = t.report(0, "test");
        assert!(r.starts_with("test: 3 distinct"), "{r}");
        assert!(r.contains("    1.000s       5  not implemented  d3d12_xs.dll!device slot 55  <-  title.exe+0x10"), "{r}");
        assert!(r.contains("most called:\n          5  d3d12_xs.dll!device slot 55"), "{r}");
    }

    #[test]
    fn long_names_are_cut_and_still_found() {
        let t = table();
        let long = "é".repeat(200);
        assert!(t.record(Kind::Stub, "m", &long, 0, String::new));
        assert!(!t.record(Kind::Stub, "m", &long, 0, String::new));
        let s = t.snapshot();
        assert_eq!(s[0].count, 2);
        assert!(s[0].name.len() < TEXT_WORDS * 8 && s[0].name.starts_with('é'));
    }

    #[test]
    fn a_full_table_counts_overflow() {
        let t = table();
        for i in 0..SLOTS + 3 {
            t.record(Kind::Stub, "m", &format!("q{i}"), 0, String::new);
        }
        assert_eq!(t.entries() as usize, SLOTS);
        assert!(t.report(0, "x").contains("3 calls not counted"));
    }

    #[test]
    fn threads_count_into_one_entry() {
        let t = table();
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for i in 0..500 {
                        t.record(Kind::NotImplemented, "m", &format!("q{}", i % 5), 0, String::new);
                    }
                });
            }
        });
        let s = t.snapshot();
        assert_eq!(s.len(), 5);
        assert_eq!(s.iter().map(|e| e.count).sum::<u64>(), 4000);
    }

    #[cfg(windows)]
    #[test]
    fn two_mappings_of_one_name_share_the_table() {
        // A pid no process has here: the block is ours alone.
        let a = Shared::open(0xFFFF_FFF0).unwrap();
        let b = Shared::open(0xFFFF_FFF0).unwrap();
        a.table().record(Kind::Error, "a.dll", "X", 0, caller_outside_this_module);
        b.table().record(Kind::Error, "a.dll", "X", 0, String::new);
        let s = b.table().snapshot();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].count, 2);
        // The test executable holds this crate: the caller is the module that started the thread.
        let exe = std::env::current_exe().unwrap();
        let exe = exe.file_name().unwrap().to_string_lossy().to_string();
        assert!(!s[0].caller.starts_with(&exe) && s[0].caller.contains("+0x"), "{}", s[0].caller);
    }

    #[test]
    fn the_report_goes_next_to_the_session_log() {
        assert_eq!(report_path(Some("x.txt".into()), Some(r"C:\l\a.log".into())), std::path::PathBuf::from("x.txt"));
        assert_eq!(report_path(None, Some(r"C:\l\a.log".into())), std::path::Path::new(r"C:\l").join(FILE_NAME));
        assert_eq!(report_path(None, Some("a.log".into())), std::path::PathBuf::from(FILE_NAME));
        assert_eq!(report_path(None, None), std::path::PathBuf::from(FILE_NAME));
    }
}
