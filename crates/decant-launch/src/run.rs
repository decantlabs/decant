//! Running the title under the Windows debugging API, which reports each DLL the loader maps (with
//! the file it came from), each `OutputDebugString` (where the shims log), each exception and the
//! exit code: the whole start of a title, without touching its code.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// What the log is told, as plain lines.
pub trait Sink {
    fn line(&mut self, text: &str);
}

impl Sink for Vec<String> {
    fn line(&mut self, text: &str) {
        self.push(text.to_owned());
    }
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub exit_code: Option<u32>,
    /// DLLs loaded, with their full paths, in load order.
    pub dlls: Vec<PathBuf>,
    pub timed_out: bool,
    /// Stopped because the caller raised its cancel flag.
    pub cancelled: bool,
}

/// The command line for `exe` and `args`, quoted as the MSVC runtime parses it.
pub fn command_line(exe: &Path, args: &[String]) -> String {
    let quote = |s: &str| -> String {
        if !s.is_empty() && !s.contains([' ', '\t', '"']) {
            return s.to_owned();
        }
        let mut out = String::from('"');
        let mut slashes = 0;
        for c in s.chars() {
            match c {
                '\\' => slashes += 1,
                '"' => {
                    out.push_str(&"\\".repeat(2 * slashes + 1));
                    out.push('"');
                    slashes = 0;
                }
                _ => {
                    out.push_str(&"\\".repeat(slashes));
                    out.push(c);
                    slashes = 0;
                }
            }
        }
        out.push_str(&"\\".repeat(2 * slashes));
        out.push('"');
        out
    };
    std::iter::once(quote(&exe.display().to_string())).chain(args.iter().map(|a| quote(a))).collect::<Vec<_>>().join(" ")
}

/// Exception codes worth a name in the log.
pub fn exception_name(code: u32) -> &'static str {
    match code {
        0x8000_0003 => "breakpoint",
        0x4000_001F => "WOW64 breakpoint",
        0xC000_0005 => "access violation",
        0xC000_001D => "illegal instruction",
        0xC000_0094 => "integer divide by zero",
        0xC000_00FD => "stack overflow",
        0xC000_0135 => "DLL not found",
        0xC000_0139 => "entry point not found",
        0xC000_0142 => "DLL initialisation failed",
        0xC000_0409 => "stack buffer overrun (fail fast)",
        0xE06D_7363 => "C++ exception",
        0x406D_1388 => "thread name",
        _ => "",
    }
}

/// Environment variable of the launcher: `0` (or `off`, `false`, `no`) turns the redirect of
/// `xgameruntime.dll` off.
pub const REDIRECT_ENV: &str = "DECANT_XGR_REDIRECT";

/// Whether the redirect is on for this value of [`REDIRECT_ENV`] (unset means on).
pub fn redirect_enabled(value: Option<&str>) -> bool {
    !value.is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"))
}

/// Whether a `LoadLibraryExW` of `requested` must be sent to `ours`: the file name is
/// `xgameruntime.dll` (a bare name or any path, either separator, any case) and the request is not
/// already `ours`.
pub fn is_runtime_request(requested: &str, ours: &Path) -> bool {
    let name = requested.rsplit(['\\', '/']).next().unwrap_or(requested);
    name.eq_ignore_ascii_case("xgameruntime.dll") && !requested.eq_ignore_ascii_case(&ours.display().to_string())
}

/// First chance access violations described in full (address, memory, registers, stack). The later ones are
/// only counted, with one line every `FIRST_CHANCE_AV_SUMMARY`. Unhandled ones are always described.
const FIRST_CHANCE_AV_DETAIL: u64 = 32;
/// See `FIRST_CHANCE_AV_DETAIL`.
const FIRST_CHANCE_AV_SUMMARY: u64 = 10_000;

/// A kernelbase file function the debugger watches for the game drive: its path argument is the first
/// one (rcx), narrow when `ansi`, and the second one (rdx) too when `second`.
pub struct FileApi {
    pub name: &'static str,
    pub ansi: bool,
    pub second: bool,
}

const fn api(name: &'static str) -> FileApi {
    FileApi { name, ansi: false, second: false }
}

/// The kernelbase functions hooked for the game drive. A first trace of a real title run (CreateFileW,
/// CreateFile2, GetFileAttributesW, GetFileAttributesExW and FindFirstFileExW all logged) showed only
/// the first two with a `G:` path, but the title imports more file functions (T22: the attribute, listing,
/// directory, delete, move calls, and DLLs of the title folder such as Bink2Scarlett.dll use CreateFile2 and
/// FindFirstFileExW), so every function that takes a path as its first argument is covered.
/// `FindFirstFileW` and `FindFirstFileExA` end in `FindFirstFileExW`, `CopyFileW` and the like in
/// `CreateFile2`; the Nt functions (`NtCreateFile`, `NtOpenFile`) take a counted string in an
/// `OBJECT_ATTRIBUTES` and are not covered: the title itself imports none of them, only
/// `xgameruntime.dll` uses `NtOpenFile` (T22).
pub static FILE_APIS: [FileApi; 12] = [
    api("CreateFileW"),
    api("GetFileAttributesW"),
    api("GetFileAttributesExW"),
    api("FindFirstFileExW"),
    api("CreateFile2"),
    FileApi { name: "GetFileAttributesA", ansi: true, second: false },
    FileApi { name: "CreateFileA", ansi: true, second: false },
    api("CreateDirectoryW"),
    api("DeleteFileW"),
    api("RemoveDirectoryW"),
    api("SetFileAttributesW"),
    FileApi { name: "MoveFileExW", ansi: false, second: true },
];

/// Lines about file calls in the log: the paths on the game drive, and the others (the system's own
/// files, mostly), each distinct (function, path) once. At the cap one line says so.
const FILE_LOG_GAME_MAX: usize = 4000;
const FILE_LOG_OTHER_MAX: usize = 400;

/// The path a call for `requested` must use instead when it starts with `letter:/` or `letter:\`
/// (any case of the letter): `folder` and the rest with backslashes. `None` for every other path.
/// A result of 260 units or more gets the `\\?\` prefix, which Win32 file calls need for it.
/// A path already inside `folder` is left alone: the overlay itself can live on that drive.
/// Every other path on the drive is redirected, the title's own `G:\pack\...` and `G://settings/...`
/// included, unless it is on the keep list (see [`game_drive_keep`]): the user's own files on a real
/// `G:` drive, such as vkd3d-proton's scratch files and the folders the launcher itself passes to the
/// layers (shader cache, probe dump).
pub fn redirect_game_path(requested: &str, letter: char, folder: &str, keep: &[String]) -> Option<String> {
    match classify_game_path(requested, letter, folder, keep) {
        Verdict::Redirect(path) => Some(path),
        _ => None,
    }
}

/// The prefixes that make a path a device path with the drive right after them (compared once `/` is `\`).
const DEVICE_PREFIXES: [&str; 3] = [r"\\?\", r"\\.\", r"\??\"];

/// What the debugger does with one path of a file call (see [`redirect_game_path`] for the rules).
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The call is sent to this path.
    Redirect(String),
    /// A path on the game drive that is already inside the title folder.
    Inside,
    /// A path on the game drive left as it is, with the reason.
    Kept(&'static str),
    /// Not a path on the game drive.
    NotGameDrive,
}

/// [`redirect_game_path`] with the reason when the path is left alone.
pub fn classify_game_path(requested: &str, letter: char, folder: &str, keep: &[String]) -> Verdict {
    // The same drive path in its Win32 extended (`\\?\G:\x`), device (`\\.\G:\x`) and NT (`\??\G:\x`) forms,
    // with either separator, is the plain `G:\x`.
    let normal = requested.replace('/', "\\");
    let body = DEVICE_PREFIXES.iter().find_map(|p| normal.strip_prefix(p)).unwrap_or(&normal);
    let mut chars = body.chars();
    let (Some(first), Some(colon), Some(sep)) = (chars.next(), chars.next(), chars.next()) else { return Verdict::NotGameDrive };
    if !first.eq_ignore_ascii_case(&letter) || colon != ':' || sep != '\\' {
        return Verdict::NotGameDrive;
    }
    // A path already inside the title folder stays as it is: when that folder is itself on the game
    // drive letter (G: is a real drive), rewriting it again would nest the folder inside itself.
    let base = folder.trim_end_matches(['\\', '/']).replace('/', "\\").to_ascii_lowercase();
    let whole = body.to_ascii_lowercase();
    if !base.is_empty() && (whole == base || whole.starts_with(&format!("{base}\\"))) {
        return Verdict::Inside;
    }
    let rest = chars.as_str();
    // The path below the drive, lowercased, one separator between names, for the keep list.
    let tail: String = rest.split('\\').filter(|c| !c.is_empty()).map(str::to_lowercase).collect::<Vec<_>>().join("\\");
    if keep.iter().any(|k| tail == k.trim_start_matches('\\') || tail.starts_with(&format!("{}\\", k.trim_start_matches('\\')))) {
        return Verdict::Kept("on the keep list of the game drive");
    }
    let mut out = format!("{}\\{}", folder.trim_end_matches(['\\', '/']), rest.trim_start_matches('\\'));
    if out.encode_utf16().count() >= 260 && !out.starts_with(r"\\") {
        out.insert_str(0, r"\\?\");
    }
    Verdict::Redirect(out)
}

/// Environment variable with more folders to keep, separated by `;` (`G:\a\b`, `\a\b` or `a\b`).
pub const KEEP_ENV: &str = "DECANT_GAME_DRIVE_KEEP";

/// One keep entry as `name\name` (lowercase, no drive, no outer separators), `None` when it is empty
/// (an entry for the whole drive would switch the redirect off), on another drive or not a path.
fn keep_entry(entry: &str, letter: char) -> Option<String> {
    let normal = entry.trim().replace('/', "\\");
    let body = DEVICE_PREFIXES.iter().find_map(|p| normal.strip_prefix(p)).unwrap_or(&normal);
    let mut chars = body.chars();
    let below = match (chars.next(), chars.next()) {
        (Some(d), Some(':')) if d.is_ascii_alphabetic() => {
            if !d.eq_ignore_ascii_case(&letter) {
                return None;
            }
            &body[2..]
        }
        _ => body,
    };
    let kept: String = below.split('\\').filter(|c| !c.is_empty()).map(str::to_lowercase).collect::<Vec<_>>().join("\\");
    (!kept.is_empty()).then_some(kept)
}

/// The folders of the game drive the debugger leaves alone: the `;` list of
/// [`KEEP_ENV`], and every folder on the drive that an environment variable of the launcher named
/// `DECANT*` carries (the shader cache of `DECANT_D3D12X_DXIL_OVERRIDE`, the probe dump folder, ...): the
/// layers read and write those at the real path, where the title never looks.
pub fn game_drive_keep(letter: char, vars: &[(String, String)]) -> Vec<String> {
    let mut keep: Vec<String> = Vec::new();
    for (name, value) in vars {
        let upper = name.to_ascii_uppercase();
        if upper == KEEP_ENV {
            keep.extend(value.split(';').filter_map(|e| keep_entry(e, letter)));
        } else if upper.starts_with("DECANT") {
            keep.extend(value.split(';').filter(|e| e.trim().chars().nth(1) == Some(':') && e.trim().starts_with(|c: char| c.eq_ignore_ascii_case(&letter))).filter_map(|e| keep_entry(e, letter)));
        }
    }
    keep.sort();
    keep.dedup();
    keep
}

/// The module whose base is the highest one at or below `address`, as `name+offset`.
pub fn locate(modules: &[(usize, String)], address: usize) -> String {
    match modules.iter().filter(|(base, _)| *base <= address).max_by_key(|(base, _)| *base) {
        Some((base, name)) => format!("{name}+{:#x}", address - base),
        None => format!("{address:#x}"),
    }
}

// The Windows debugging API is only reachable through FFI: the unsafe blocks are confined to this
// module, each with what makes it sound.
#[cfg(windows)]
#[allow(unsafe_code)]
mod platform {
    use super::*;
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{CloseHandle, DBG_CONTINUE, DBG_EXCEPTION_NOT_HANDLED, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        CREATE_PROCESS_DEBUG_EVENT, CREATE_THREAD_DEBUG_EVENT, ContinueDebugEvent, DEBUG_EVENT, EXCEPTION_DEBUG_EVENT, EXIT_PROCESS_DEBUG_EVENT, EXIT_THREAD_DEBUG_EVENT,
        LOAD_DLL_DEBUG_EVENT, OUTPUT_DEBUG_STRING_EVENT, RIP_EVENT, ReadProcessMemory, UNLOAD_DLL_DEBUG_EVENT, WaitForDebugEvent,
    };
    use windows_sys::Win32::System::Threading::{CREATE_UNICODE_ENVIRONMENT, CreateProcessW, DEBUG_ONLY_THIS_PROCESS, PROCESS_INFORMATION, STARTUPINFOW, TerminateProcess};

    /// Breaks on the entry of kernelbase functions, run by the debugger (one int3 per function, one
    /// single step to run the replaced instruction, then the break is put back):
    /// `LoadLibraryExW` (what `LoadLibraryW`, `LoadLibraryA` and `LoadLibraryExA` all end in) is
    /// pointed at the interposer of the title folder when the title asks for `xgameruntime.dll`, and
    /// the file functions in [`FILE_APIS`] get a path under the game drive `G:/` rewritten to the
    /// title folder when the drive cannot be mapped. All of them take the path as first argument (rcx).
    #[cfg(target_arch = "x86_64")]
    mod hook {
        use super::*;
        use std::collections::HashMap;
        use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64, FlushInstructionCache, GetThreadContext, SetThreadContext, WriteProcessMemory};
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAllocEx};
        use windows_sys::Win32::System::Threading::{OpenThread, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT};

        /// `CONTEXT` must sit on a 16 byte boundary, which windows-sys does not declare.
        #[repr(C, align(16))]
        struct Aligned(CONTEXT);

        pub const SINGLE_STEP: u32 = 0x8000_0004;
        const TRAP_FLAG: u32 = 0x100;
        const LOAD_LIBRARY: &str = "LoadLibraryExW";
        /// Size of one block of the path arena in the debuggee.
        const ARENA_BLOCK: usize = 1 << 20;
        /// The longest path read from the debuggee, in UTF-16 units.
        const MAX_PATH_UNITS: usize = 32768;

        /// One armed function.
        struct Target {
            name: &'static str,
            address: usize,
            original: u8,
            /// The threads that run the restored instruction. Two threads can stop on the same break
            /// before it is lifted (their events queue up), so each one is kept, and the break is put
            /// back once the last of them has stepped (a single slot lost the first thread's step,
            /// which then reached the title as an unhandled single step at LoadLibraryExW+5).
            stepping: Vec<u32>,
        }

        /// What the file functions rewrite: `letter:/` becomes `folder`.
        pub struct Drive {
            pub letter: char,
            pub folder: String,
            /// The folders of the drive left alone (see `game_drive_keep`), read once at setup.
            pub keep: Vec<String>,
        }

        pub struct Hook {
            targets: Vec<Target>,
            /// The interposer, and its path in the debuggee once written.
            runtime: Option<(PathBuf, Option<usize>)>,
            drive: Option<Drive>,
            /// Rewritten paths in the debuggee, by the path asked for (None: not rewritten or could not
            /// be written). Narrow and wide copies are kept apart.
            rewritten: HashMap<(String, bool), (Option<usize>, bool)>,
            /// The block new paths are carved from: (address, bytes used).
            arena: Option<(usize, usize)>,
            /// The (function, path) pairs already logged.
            seen: HashSet<(&'static str, String)>,
            /// Lines written for the game drive and for the other paths (see the caps).
            logged: (usize, usize),
            /// Per function: calls, rewritten calls, calls on the game drive left as they were.
            counts: HashMap<&'static str, [u64; 3]>,
        }

        fn write(process: HANDLE, address: usize, bytes: &[u8]) -> bool {
            let mut written = 0usize;
            // SAFETY: writes `bytes.len()` bytes from a live slice.
            let ok = unsafe { WriteProcessMemory(process, address as *const c_void, bytes.as_ptr().cast(), bytes.len(), &mut written) } != 0 && written == bytes.len();
            // SAFETY: flushes the range just written.
            unsafe { FlushInstructionCache(process, address as *const c_void, bytes.len()) };
            ok
        }

        fn read(process: HANDLE, address: usize, buffer: &mut [u8]) -> usize {
            let mut read = 0usize;
            // SAFETY: reads at most the buffer's length into it.
            unsafe { ReadProcessMemory(process, address as *const c_void, buffer.as_mut_ptr().cast(), buffer.len(), &mut read) };
            read
        }

        /// The NUL-terminated UTF-16 string at `address` (at most [`MAX_PATH_UNITS`] units), read
        /// page by page so that a string at the end of a mapping is still read.
        fn read_wide(process: HANDLE, address: usize) -> String {
            let mut units: Vec<u16> = Vec::new();
            let mut at = address;
            'pages: while units.len() < MAX_PATH_UNITS {
                let room = (0x1000 - (at & 0xFFF)) & !1;
                let mut bytes = vec![0u8; room.max(2)];
                let n = read(process, at, &mut bytes);
                if n < 2 {
                    break;
                }
                for c in bytes[..n].as_chunks::<2>().0 {
                    let u = u16::from_le_bytes(*c);
                    if u == 0 {
                        break 'pages;
                    }
                    units.push(u);
                }
                at += n & !1;
            }
            String::from_utf16_lossy(&units)
        }

        /// The NUL-terminated narrow string at `address` (at most [`MAX_PATH_UNITS`] bytes), read page by
        /// page like [`read_wide`]; a byte above 127 shows as U+FFFD (the code page is not known here).
        fn read_narrow(process: HANDLE, address: usize) -> String {
            let mut bytes: Vec<u8> = Vec::new();
            let mut at = address;
            'pages: while bytes.len() < MAX_PATH_UNITS {
                let room = 0x1000 - (at & 0xFFF);
                let mut chunk = vec![0u8; room];
                let n = read(process, at, &mut chunk);
                if n == 0 {
                    break;
                }
                for &b in &chunk[..n] {
                    if b == 0 {
                        break 'pages;
                    }
                    bytes.push(b);
                }
                at += n;
            }
            String::from_utf8_lossy(&bytes).into_owned()
        }

        impl Hook {
            /// Arms the breaks in the `kernelbase.dll` mapped at `base`: `LoadLibraryExW` when there
            /// is an interposer `runtime`, the [`FILE_APIS`] when there is a `drive` to redirect. The
            /// offsets are taken from this process's copy of the same file.
            pub fn arm(process: HANDLE, base: usize, runtime: Option<PathBuf>, drive: Option<Drive>) -> Result<Hook, String> {
                let module: Vec<u16> = "kernelbase.dll\0".encode_utf16().collect();
                // SAFETY: a NUL-terminated name; the module stays loaded in this process.
                let mine = unsafe { GetModuleHandleW(module.as_ptr()) };
                if mine.is_null() {
                    return Err("no kernelbase.dll in this process".into());
                }
                let mut names: Vec<&'static str> = Vec::new();
                if runtime.is_some() {
                    names.push(LOAD_LIBRARY);
                }
                if drive.is_some() {
                    names.extend(FILE_APIS.iter().map(|a| a.name));
                }
                let mut hook = Hook {
                    targets: Vec::new(),
                    runtime: runtime.map(|p| (p, None)),
                    drive,
                    rewritten: HashMap::new(),
                    arena: None,
                    seen: HashSet::new(),
                    logged: (0, 0),
                    counts: HashMap::new(),
                };
                // SAFETY: this process's own mapping of kernelbase.dll: the DOS header's `e_lfanew` (at 0x3C)
                // gives the NT headers, whose optional header holds `SizeOfImage` (at +0x50).
                let image_size = unsafe {
                    let image = mine as *const u8;
                    let nt = image.add(image.add(0x3C).cast::<u32>().read_unaligned() as usize);
                    nt.add(0x50).cast::<u32>().read_unaligned() as usize
                };
                for name in names {
                    let cname = format!("{name}\0");
                    // SAFETY: `cname` is NUL-terminated and `mine` a loaded module.
                    let entry = unsafe { GetProcAddress(mine, cname.as_ptr()) }.map(|f| f as usize).ok_or(format!("{name} not found in this process's kernelbase.dll"))?;
                    // An export that forwards to another DLL answers with that DLL's address: not ours to break.
                    if entry < mine as usize || entry >= mine as usize + image_size {
                        return Err(format!("{name} is not in this process's kernelbase.dll (a forwarder)"));
                    }
                    let address = base + (entry - mine as usize);
                    // Two names at one address share one break (the second would save the 0xCC as its byte).
                    if hook.targets.iter().any(|t| t.address == address) {
                        continue;
                    }
                    let mut original = [0u8; 1];
                    if read(process, address, &mut original) != 1 {
                        return Err(format!("cannot read {name} at {address:#x}"));
                    }
                    if !write(process, address, &[0xCC]) {
                        return Err(format!("cannot write {name} at {address:#x}"));
                    }
                    hook.targets.push(Target { name, address, original: original[0], stepping: Vec::new() });
                }
                Ok(hook)
            }

            /// The armed functions as `name at address`, for the log.
            pub fn describe(&self) -> String {
                self.targets.iter().map(|t| format!("{} at {:#x}", t.name, t.address)).collect::<Vec<_>>().join(", ")
            }

            pub fn owns(&self, address: usize) -> bool {
                self.targets.iter().any(|t| t.address == address)
            }

            /// A break on an entry, `thread` stopped there: reports the call and redirects it when it
            /// asks for the system runtime or a path under the game drive. The first instruction is
            /// then run with the original byte, and [`Hook::stepped`] puts the break back.
            pub fn hit(&mut self, process: HANDLE, thread: u32, address: usize, modules: &[(usize, String)], log: &mut dyn FnMut(&str)) {
                let Some(index) = self.targets.iter().position(|t| t.address == address) else { return };
                let (name, original) = (self.targets[index].name, self.targets[index].original);
                // SAFETY: a plain handle request.
                let handle = unsafe { OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, 0, thread) };
                if handle.is_null() {
                    write(process, address, &[original]);
                    log(&format!("hook: cannot open the thread, {name} hook removed"));
                    return;
                }
                // SAFETY: a plain structure the call fills; the flags name the parts wanted.
                let mut aligned: Aligned = unsafe { std::mem::zeroed() };
                let context = &mut aligned.0;
                context.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
                // SAFETY: `handle` is open with the rights the call needs.
                if unsafe { GetThreadContext(handle, context) } != 0 {
                    if name == LOAD_LIBRARY {
                        self.redirect_runtime(process, context, modules, log);
                    } else if let Some(api) = FILE_APIS.iter().find(|a| a.name == name) {
                        self.redirect_drive(process, context, api, log);
                    }
                    // Run the original first instruction, and break again after it.
                    context.Rip = address as u64;
                    context.EFlags |= TRAP_FLAG;
                    write(process, address, &[original]);
                    if !self.targets[index].stepping.contains(&thread) {
                        self.targets[index].stepping.push(thread);
                    }
                    // SAFETY: `handle` is open with the rights the call needs.
                    unsafe { SetThreadContext(handle, context) };
                } else {
                    write(process, address, &[original]);
                    log(&format!("hook: cannot read the thread context ({}), {name} hook removed", std::io::Error::last_os_error()));
                }
                // SAFETY: the handle is ours.
                unsafe { CloseHandle(handle) };
            }

            fn redirect_runtime(&mut self, process: HANDLE, context: &mut CONTEXT, modules: &[(usize, String)], log: &mut dyn FnMut(&str)) {
                let Some((ours, _)) = self.runtime.clone() else { return };
                let requested = read_wide(process, context.Rcx as usize);
                let mut ret = [0u8; 8];
                read(process, context.Rsp as usize, &mut ret);
                let from = locate(modules, usize::from_le_bytes(ret));
                // The interposer loads the real runtime by this very call: not redirected.
                if is_runtime_request(&requested, &ours) && !from.to_ascii_lowercase().starts_with("xgameruntime.dll+") {
                    let flags = context.R8 as u32;
                    if let Some(remote) = self.remote_runtime(process) {
                        context.Rcx = remote as u64;
                        log(&format!("redirect LoadLibraryExW(\"{requested}\", flags {flags:#x}) from {from} to {}", ours.display()));
                    } else {
                        log(&format!("LoadLibraryExW(\"{requested}\", flags {flags:#x}) from {from}: cannot write the redirect path"));
                    }
                }
            }

            /// Points the first argument at a copy of the path inside the title folder when it
            /// starts with the game drive. The copy lives in the debuggee for the rest of the run,
            /// one per distinct path, so no thread ever reads a buffer that is written again.
            fn redirect_drive(&mut self, process: HANDLE, context: &mut CONTEXT, api: &'static FileApi, log: &mut dyn FnMut(&str)) {
                if let Some(remote) = self.redirect_path(process, context.Rcx as usize, api, true, log) {
                    context.Rcx = remote as u64;
                }
                // `MoveFileExW(existing, new, flags)`: the second path is rdx.
                if let Some(remote) = api.second.then(|| self.redirect_path(process, context.Rdx as usize, api, false, log)).flatten() {
                    context.Rdx = remote as u64;
                }
            }

            /// The pointer to the rewritten copy of the path at `at` when it is on the game drive. `first`:
            /// the call is counted with this path (a function with two paths is one call).
            fn redirect_path(&mut self, process: HANDLE, at: usize, api: &'static FileApi, first: bool, log: &mut dyn FnMut(&str)) -> Option<usize> {
                let drive = self.drive.as_ref()?;
                let (letter, folder, keep) = (drive.letter, drive.folder.clone(), drive.keep.clone());
                if at == 0 {
                    return None;
                }
                let requested = if api.ansi { read_narrow(process, at) } else { read_wide(process, at) };
                let key = (requested.clone(), api.ansi);
                // The verdict is worked out once per path: the pointer to the rewritten copy (if any), and
                // whether the path is on the game drive at all.
                let (remote, on_drive) = match self.rewritten.get(&key) {
                    Some(&known) => known,
                    None => {
                        let verdict = classify_game_path(&requested, letter, &folder, &keep);
                        let remote = match &verdict {
                            Verdict::Redirect(new) if api.ansi && !new.is_ascii() => None,
                            Verdict::Redirect(new) if api.ansi => self.carve(process, &[new.as_bytes(), &[0]].concat()),
                            Verdict::Redirect(new) => self.carve(process, &new.encode_utf16().chain(std::iter::once(0)).flat_map(u16::to_le_bytes).collect::<Vec<u8>>()),
                            _ => None,
                        };
                        let known = (remote, verdict != Verdict::NotGameDrive);
                        self.rewritten.insert(key, known);
                        known
                    }
                };
                let calls = self.counts.entry(api.name).or_default();
                calls[0] += u64::from(first);
                calls[1] += u64::from(remote.is_some());
                calls[2] += u64::from(on_drive && remote.is_none());
                if self.seen.insert((api.name, requested.clone())) {
                    // A line for each (function, path) once, from the verdict again (cheap, once per pair).
                    let name = api.name;
                    let (line, game) = match classify_game_path(&requested, letter, &folder, &keep) {
                        Verdict::Redirect(new) if remote.is_some() => (format!("redirect {name}(\"{requested}\") to {new}"), true),
                        Verdict::Redirect(_) => (format!("{name}(\"{requested}\"): cannot write the redirect path"), true),
                        Verdict::Inside => (format!("no redirect {name}(\"{requested}\"): already in the title folder"), true),
                        Verdict::Kept(why) => (format!("no redirect {name}(\"{requested}\"): {why}"), true),
                        Verdict::NotGameDrive => (format!("file {name}(\"{requested}\")"), false),
                    };
                    let (count, max) = if game { (&mut self.logged.0, FILE_LOG_GAME_MAX) } else { (&mut self.logged.1, FILE_LOG_OTHER_MAX) };
                    if *count < max {
                        *count += 1;
                        log(&line);
                        if *count == max {
                            log(&format!("file calls: {max} lines {}, no more of them (the counts come at the end)", if game { "on the game drive" } else { "for other paths" }));
                        }
                    }
                }
                remote
            }

            /// The lines for the end of the run: the calls of each file function and what was done with them.
            pub fn summary(&self) -> Vec<String> {
                let mut names: Vec<&&'static str> = self.counts.keys().collect();
                names.sort();
                names
                    .into_iter()
                    .map(|n| {
                        let [calls, rewritten, kept] = self.counts[*n];
                        format!("file calls: {n} {calls} (rewritten {rewritten}, on the game drive and left as they were {kept})")
                    })
                    .collect()
            }

            /// `bytes` copied into the arena of the debuggee (a new block when the current is full).
            fn carve(&mut self, process: HANDLE, bytes: &[u8]) -> Option<usize> {
                let size = bytes.len().next_multiple_of(8);
                if self.arena.is_none_or(|(_, used)| used + size > ARENA_BLOCK) {
                    let block = size.max(ARENA_BLOCK);
                    // SAFETY: a fresh read/write block in the debuggee.
                    let at = unsafe { VirtualAllocEx(process, std::ptr::null(), block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as usize;
                    if at == 0 {
                        return None;
                    }
                    self.arena = Some((at, 0));
                }
                let (base, used) = self.arena?;
                let at = base + used;
                if !write(process, at, bytes) {
                    return None;
                }
                self.arena = Some((base, used + size));
                Some(at)
            }

            /// The single step after [`Hook::hit`]: true when it was ours (the break is back).
            pub fn stepped(&mut self, process: HANDLE, thread: u32) -> bool {
                let Some(t) = self.targets.iter_mut().find(|t| t.stepping.contains(&thread)) else { return false };
                t.stepping.retain(|&s| s != thread);
                if t.stepping.is_empty() {
                    write(process, t.address, &[0xCC]);
                }
                true
            }

            fn remote_runtime(&mut self, process: HANDLE) -> Option<usize> {
                let (ours, remote) = self.runtime.clone()?;
                if remote.is_none() {
                    let bytes: Vec<u8> = ours.as_os_str().encode_wide().chain(std::iter::once(0)).flat_map(u16::to_le_bytes).collect();
                    let at = self.carve(process, &bytes);
                    self.runtime = Some((ours, at));
                    return at;
                }
                remote
            }
        }
    }

    /// The thread sampler of [`crate::sampler`]: the debuggee's threads and modules as the debug
    /// events announce them, and a sampling round whenever one is due.
    #[cfg(target_arch = "x86_64")]
    mod sampling {
        use super::*;
        use crate::sampler::{Module, Profile};
        use std::collections::HashMap;
        use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, GetThreadContext};
        use windows_sys::Win32::System::Threading::{ResumeThread, SuspendThread};

        /// `CONTEXT` must sit on a 16 byte boundary, which windows-sys does not declare.
        #[repr(C, align(16))]
        struct Aligned(CONTEXT);

        const REPORT_EVERY: Duration = Duration::from_secs(30);

        pub struct Sampler {
            pub every: Duration,
            next: Instant,
            report_at: Instant,
            started: Instant,
            threads: HashMap<u32, HANDLE>,
            modules: Vec<Module>,
            profile: Profile,
            out: PathBuf,
        }

        fn read(process: HANDLE, address: usize, buffer: &mut [u8]) -> usize {
            let mut got = 0usize;
            // SAFETY: reads at most the buffer's length into it.
            unsafe { ReadProcessMemory(process, address as *const c_void, buffer.as_mut_ptr().cast(), buffer.len(), &mut got) };
            got
        }

        fn u32_at(process: HANDLE, address: usize) -> Option<u32> {
            let mut b = [0u8; 4];
            (read(process, address, &mut b) == 4).then(|| u32::from_le_bytes(b))
        }

        impl Sampler {
            pub fn new(every: Duration, out: PathBuf) -> Self {
                let now = Instant::now();
                let _ = std::fs::write(&out, format!("thread samples every {} ms\n", every.as_millis()));
                Sampler { every, next: now, report_at: now + REPORT_EVERY, started: now, threads: HashMap::new(), modules: Vec::new(), profile: Profile::default(), out }
            }

            pub fn thread(&mut self, tid: u32, handle: HANDLE) {
                self.threads.insert(tid, handle);
            }

            pub fn thread_gone(&mut self, tid: u32) {
                self.threads.remove(&tid);
            }

            /// A module at `base`: its size is `SizeOfImage` of its PE header in the debuggee.
            pub fn module(&mut self, process: HANDLE, base: usize, name: String) {
                let size = u32_at(process, base + 0x3C).and_then(|pe| u32_at(process, base + pe as usize + 0x50)).unwrap_or(0);
                self.modules.push((base, size as usize, name));
            }

            pub fn unload(&mut self, base: usize) {
                self.modules.retain(|(b, _, _)| *b != base);
            }

            /// One round when it is due, and the report when that is due.
            pub fn poll(&mut self, process: HANDLE) {
                let now = Instant::now();
                if now < self.next {
                    return;
                }
                self.next = now + self.every;
                for (&tid, &handle) in &self.threads {
                    // SAFETY: `handle` is the debuggee thread handle the debug event gave us, alive
                    // until its exit event. The thread is resumed right after the context is read.
                    unsafe {
                        if SuspendThread(handle) == u32::MAX {
                            continue;
                        }
                        let mut aligned: Aligned = std::mem::zeroed();
                        aligned.0.ContextFlags = CONTEXT_CONTROL_AMD64;
                        let ok = GetThreadContext(handle, &mut aligned.0) != 0;
                        let mut bytes = [0u8; 8 * 64];
                        let got = if ok { read(process, aligned.0.Rsp as usize, &mut bytes) } else { 0 };
                        ResumeThread(handle);
                        if !ok {
                            continue;
                        }
                        let words: Vec<usize> = bytes[..got].as_chunks::<8>().0.iter().map(|w| u64::from_le_bytes(*w) as usize).collect();
                        self.profile.add(tid, crate::sampler::key(&self.modules, aligned.0.Rip as usize, &words));
                    }
                }
                self.profile.tick();
                if now >= self.report_at {
                    self.report_at = now + REPORT_EVERY;
                    self.write_report();
                }
            }

            pub fn write_report(&mut self) {
                use std::io::Write;
                let text = format!("--- at {:.1} s\n{}", self.started.elapsed().as_secs_f64(), self.profile.take_report(40));
                if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&self.out) {
                    let _ = f.write_all(text.as_bytes());
                }
            }
        }
    }

    /// The first `take` of the `words` stack words of `thread` that fall inside the title image (the
    /// first module), likely return addresses, as `module+offset` joined by spaces. `None` when there
    /// is none.
    #[cfg(target_arch = "x86_64")]
    fn title_calls(process: HANDLE, thread: u32, modules: &[(usize, String)], words: usize, take: usize) -> Option<String> {
        let title = modules.first().map(|(base, _)| *base).unwrap_or(0);
        let calls: Vec<String> = crate::cpu_emu::platform::stack_words(process, thread, words)
            .into_iter()
            .filter(|w| title != 0 && (title..title + 0x1000_0000).contains(w))
            .take(take)
            .map(|w| locate(modules, w))
            .collect();
        (!calls.is_empty()).then(|| calls.join(" "))
    }

    fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    fn file_path(file: HANDLE) -> Option<PathBuf> {
        if file.is_null() {
            return None;
        }
        let mut buffer = vec![0u16; 1024];
        // SAFETY: the buffer is as long as the length passed.
        let n = unsafe { GetFinalPathNameByHandleW(file, buffer.as_mut_ptr(), buffer.len() as u32, 0) } as usize;
        // SAFETY: the debugger owns the handle the event gave it.
        unsafe { CloseHandle(file) };
        let text = String::from_utf16_lossy(buffer.get(..n)?);
        Some(PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text)))
    }

    fn debug_string(process: HANDLE, at: *const c_void, length: u16, unicode: bool) -> String {
        let bytes = usize::from(length) * if unicode { 2 } else { 1 };
        let mut buffer = vec![0u8; bytes];
        let mut read = 0usize;
        // SAFETY: reads at most `bytes` into a buffer of that size.
        if unsafe { ReadProcessMemory(process, at, buffer.as_mut_ptr().cast(), bytes, &mut read) } == 0 {
            return String::new();
        }
        buffer.truncate(read);
        let text = if unicode {
            String::from_utf16_lossy(&buffer.as_chunks::<2>().0.iter().map(|&c| u16::from_le_bytes(c)).collect::<Vec<_>>())
        } else {
            String::from_utf8_lossy(&buffer).into_owned()
        };
        text.trim_end_matches(['\0', '\r', '\n']).to_owned()
    }

    /// The environment of this process plus `extra`, as a sorted UTF-16 block.
    ///
    /// The title runs under this debugger, so Windows gives every heap of it the debug heap
    /// (`NtGlobalFlag` checks, no low fragmentation heap) unless `_NO_DEBUG_HEAP=1` is set: then
    /// every allocation of the layer's shader translator waited on the heap lock (a retail title, one
    /// run: a translation that takes a few ms alone took a hundred times longer in the title).
    /// It is set unless the environment names it already (`_NO_DEBUG_HEAP=0` keeps the debug heap).
    fn environment(extra: &[(String, String)]) -> Vec<u16> {
        let mut vars: Vec<(String, String)> = std::env::vars_os().map(|(k, v)| (k.to_string_lossy().into_owned(), v.to_string_lossy().into_owned())).collect();
        for (k, v) in extra {
            vars.retain(|(key, _)| !key.eq_ignore_ascii_case(k));
            vars.push((k.clone(), v.clone()));
        }
        if !vars.iter().any(|(k, _)| k.eq_ignore_ascii_case("_NO_DEBUG_HEAP")) {
            vars.push(("_NO_DEBUG_HEAP".to_string(), "1".to_string()));
        }
        vars.sort_by_key(|(k, _)| k.to_uppercase());
        let mut block: Vec<u16> = vars.iter().flat_map(|(k, v)| format!("{k}={v}").encode_utf16().chain(std::iter::once(0)).collect::<Vec<_>>()).collect();
        block.push(0);
        block
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(exe: &Path, args: &[String], folder: &Path, extra_env: &[(String, String)], timeout: Option<Duration>, cancel: &AtomicBool, redirect: bool, game_drive: Option<char>, log: &mut dyn Sink) -> std::io::Result<Outcome> {
        let start = Instant::now();
        let stamp = |start: Instant| format!("{:>8.3}", start.elapsed().as_secs_f64());
        let mut command = wide(std::ffi::OsStr::new(&command_line(exe, args)));
        let application = wide(exe.as_os_str());
        let directory = wide(folder.as_os_str());
        let env = environment(extra_env);
        // The folders of the game drive the redirect leaves alone: fixed, from the environment the title inherits.
        let env_vars: Vec<(String, String)> = std::env::vars().chain(extra_env.iter().cloned()).collect();
        let drive_keep = game_drive.map(|letter| game_drive_keep(letter, &env_vars)).unwrap_or_default();
        // SAFETY: plain structures the call fills.
        let mut startup: STARTUPINFOW = unsafe { std::mem::zeroed() };
        startup.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
        let mut info: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: every pointer is to a live, NUL-terminated buffer of this frame.
        let made = unsafe {
            CreateProcessW(
                application.as_ptr(),
                command.as_mut_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                DEBUG_ONLY_THIS_PROCESS | CREATE_UNICODE_ENVIRONMENT,
                env.as_ptr().cast(),
                directory.as_ptr(),
                &startup,
                &mut info,
            )
        };
        if made == 0 {
            return Err(std::io::Error::last_os_error());
        }
        log.line(&format!("{} start {} (process {})", stamp(start), command_line(exe, args), info.dwProcessId));
        let gentle_given = extra_env.iter().rev().find(|(k, _)| k.eq_ignore_ascii_case(crate::gentle::ENV)).map(|(_, v)| v.clone());
        let mut gentle = if crate::gentle::enabled(gentle_given.or_else(|| std::env::var(crate::gentle::ENV).ok()).as_deref()) {
            let (state, line) = crate::gentle::Gentle::apply(info.hProcess);
            log.line(&format!("{} {line}", stamp(start)));
            Some(state)
        } else {
            None
        };
        let mut outcome = Outcome { exit_code: None, dlls: Vec::new(), timed_out: false, cancelled: false };
        let mut first_breakpoint = true;
        // First chance access violations seen. Only the first ones are described in full: the dirty
        // tracking of xmem.dll faults on purpose (a video decoder thousands of times a second), its own
        // vectored handler resolves them, and the description stops the title thread for a stack walk.
        let mut first_chance_av: u64 = 0;
        // The launcher's own time for the counted ones (from the event to its continue), and when the
        // current run of `FIRST_CHANCE_AV_SUMMARY` began: the summary line gives both per event.
        let mut fast_av_micros: u128 = 0;
        let mut fast_av_since = Instant::now();
        let mut threads = 0usize;
        let ours = folder.join("xgameruntime.dll");
        let want_runtime = redirect && ours.is_file();
        let want_hook = want_runtime || game_drive.is_some();
        if !want_runtime {
            let why = if ours.is_file() { format!("{REDIRECT_ENV} is off") } else { "no xgameruntime.dll in the title folder".to_owned() };
            log.line(&format!("{} no redirect of xgameruntime.dll ({why})", stamp(start)));
        }
        #[cfg(target_arch = "x86_64")]
        let mut hook: Option<hook::Hook> = None;
        let mut modules: Vec<(usize, String)> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        let emulate_cpu = {
            // The switch can also come with the title's own environment (`extra_env`).
            let given = extra_env.iter().rev().find(|(k, _)| k.eq_ignore_ascii_case(crate::cpu_emu::ENV)).map(|(_, v)| v.clone());
            crate::cpu_emu::enabled(given.or_else(|| std::env::var(crate::cpu_emu::ENV).ok()).as_deref())
        };
        #[cfg(target_arch = "x86_64")]
        let mut cpu_stats = crate::cpu_emu::Stats::default();
        #[cfg(target_arch = "x86_64")]
        let mut sampler = {
            let given = extra_env.iter().rev().find(|(k, _)| k.eq_ignore_ascii_case(crate::sampler::ENV)).map(|(_, v)| v.clone());
            crate::sampler::interval(given.or_else(|| std::env::var(crate::sampler::ENV).ok()).as_deref()).map(|every| sampling::Sampler::new(every, folder.join("decant_sample.txt")))
        };
        // The debugging switches of docs/DEBUG_TOOLS.md (DECANT_WATCH, DECANT_STACKS, DECANT_WAITS, DECANT_CALLS).
        #[cfg(target_arch = "x86_64")]
        let mut tools = crate::tools::Tools::new(info.dwProcessId, info.hProcess, folder, extra_env, start, &mut |t: &str| log.line(&format!("{} {t}", stamp(start))));
        #[cfg(target_arch = "x86_64")]
        let wait_ms = sampler.as_ref().map_or(100, |s| s.every.as_millis().clamp(1, 100) as u32).min(tools.wait_ms().unwrap_or(100));
        #[cfg(not(target_arch = "x86_64"))]
        let wait_ms = 100;
        loop {
            #[cfg(target_arch = "x86_64")]
            if let Some(s) = sampler.as_mut() {
                s.poll(info.hProcess);
            }
            #[cfg(target_arch = "x86_64")]
            tools.poll(info.hProcess, &mut |t: &str| log.line(&format!("{} {t}", stamp(start))));
            if let Some(line) = gentle.as_mut().and_then(|g| g.poll(info.hProcess)) {
                log.line(&format!("{} {line}", stamp(start)));
            }
            if cancel.load(Ordering::Relaxed) && !outcome.cancelled {
                log.line(&format!("{} cancelled: the title is stopped", stamp(start)));
                outcome.cancelled = true;
                // SAFETY: the process handle is ours.
                unsafe { TerminateProcess(info.hProcess, 0xFFFF_FFFF) };
            }
            // Checked on every pass, not only when the wait gives up: a busy title sends debug events faster
            // than `wait_ms`, the wait never times out, and the deadline used to pass unnoticed.
            if timeout.is_some_and(|t| start.elapsed() > t) && !outcome.timed_out {
                log.line(&format!("{} timeout: the title is stopped", stamp(start)));
                outcome.timed_out = true;
                // SAFETY: the process handle is ours.
                unsafe { TerminateProcess(info.hProcess, 0xFFFF_FFFF) };
            }
            // SAFETY: filled by the call.
            let mut event: DEBUG_EVENT = unsafe { std::mem::zeroed() };
            // SAFETY: waits on this thread, which started the debuggee.
            if unsafe { WaitForDebugEvent(&mut event, wait_ms) } == 0 {
                continue;
            }
            let received = Instant::now();
            let mut status = DBG_CONTINUE;
            // A first chance access violation past the described ones goes straight back to the title, whose
            // own handler (xmem's dirty tracking) resolves it: no hook, tool or memory read on the way.
            let fast_av = event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT
                && unsafe { event.u.Exception.ExceptionRecord.ExceptionCode } as u32 == 0xC000_0005
                && unsafe { event.u.Exception.dwFirstChance } != 0
                && first_chance_av >= FIRST_CHANCE_AV_DETAIL;
            if fast_av {
                first_chance_av += 1;
                // SAFETY: the event came from WaitForDebugEvent on this thread.
                unsafe { ContinueDebugEvent(event.dwProcessId, event.dwThreadId, DBG_EXCEPTION_NOT_HANDLED) };
                fast_av_micros += received.elapsed().as_micros();
                if first_chance_av.is_multiple_of(FIRST_CHANCE_AV_SUMMARY) {
                    let span = fast_av_since.elapsed().as_secs_f64();
                    log.line(&format!(
                        "{} {first_chance_av} first chance access violations so far (the first {FIRST_CHANCE_AV_DETAIL} are described above), the last {FIRST_CHANCE_AV_SUMMARY} in {span:.1} s with {:.1} us of the launcher each",
                        stamp(start),
                        fast_av_micros as f64 / FIRST_CHANCE_AV_SUMMARY as f64
                    ));
                    fast_av_micros = 0;
                    fast_av_since = Instant::now();
                }
                continue;
            }
            // SAFETY: each arm reads the union member its event code names.
            unsafe {
                match event.dwDebugEventCode {
                    // The image is the executable started (its final path can be another name of a
                    // hard linked file).
                    CREATE_PROCESS_DEBUG_EVENT => {
                        modules.push((event.u.CreateProcessInfo.lpBaseOfImage as usize, exe.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned())));
                        #[cfg(target_arch = "x86_64")]
                        if let Some(s) = sampler.as_mut() {
                            s.thread(event.dwThreadId, event.u.CreateProcessInfo.hThread);
                            s.module(info.hProcess, event.u.CreateProcessInfo.lpBaseOfImage as usize, modules[modules.len() - 1].1.clone());
                        }
                        #[cfg(target_arch = "x86_64")]
                        {
                            let mut say = |t: &str| log.line(&format!("{} {t}", stamp(start)));
                            tools.module(info.hProcess, event.u.CreateProcessInfo.lpBaseOfImage as usize, &modules[modules.len() - 1].1, &mut say);
                            tools.thread(info.hProcess, event.dwThreadId, event.u.CreateProcessInfo.hThread, &mut say);
                        }
                        drop(file_path(event.u.CreateProcessInfo.hFile));
                    }
                    LOAD_DLL_DEBUG_EVENT => {
                        let dll = event.u.LoadDll;
                        match file_path(dll.hFile) {
                            Some(path) => {
                                let in_folder = path.starts_with(folder);
                                log.line(&format!("{} load {:#014x} {}{}", stamp(start), dll.lpBaseOfDll as usize, path.display(), if in_folder { "  [title folder]" } else { "" }));
                                modules.push((dll.lpBaseOfDll as usize, path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned())));
                                #[cfg(target_arch = "x86_64")]
                                if let Some(s) = sampler.as_mut() {
                                    s.module(info.hProcess, dll.lpBaseOfDll as usize, modules[modules.len() - 1].1.clone());
                                }
                                #[cfg(target_arch = "x86_64")]
                                tools.module(info.hProcess, dll.lpBaseOfDll as usize, &modules[modules.len() - 1].1, &mut |t: &str| log.line(&format!("{} {t}", stamp(start))));
                                #[cfg(target_arch = "x86_64")]
                                if want_hook && hook.is_none() && path.file_name().is_some_and(|n| n.eq_ignore_ascii_case("kernelbase.dll")) {
                                    match hook::Hook::arm(info.hProcess, dll.lpBaseOfDll as usize, want_runtime.then(|| ours.clone()), game_drive.map(|letter| hook::Drive { letter, folder: folder.display().to_string(), keep: drive_keep.clone() })) {
                                        Ok(h) => { log.line(&format!("{} redirect armed: {}", stamp(start), h.describe())); log.line(&format!("{} redirect keeps on the game drive: {}", stamp(start), if drive_keep.is_empty() { "nothing".to_owned() } else { drive_keep.join(" ; ") })); hook = Some(h) }
                                        Err(e) => log.line(&format!("{} redirect not armed: {e}", stamp(start))),
                                    }
                                }
                                outcome.dlls.push(path);
                            }
                            None => log.line(&format!("{} load {:#014x} (no file)", stamp(start), dll.lpBaseOfDll as usize)),
                        }
                    }
                    UNLOAD_DLL_DEBUG_EVENT => {
                        log.line(&format!("{} unload {:#014x}", stamp(start), event.u.UnloadDll.lpBaseOfDll as usize));
                        #[cfg(target_arch = "x86_64")]
                        if let Some(s) = sampler.as_mut() {
                            s.unload(event.u.UnloadDll.lpBaseOfDll as usize);
                        }
                        #[cfg(target_arch = "x86_64")]
                        tools.unload(event.u.UnloadDll.lpBaseOfDll as usize);
                    }
                    CREATE_THREAD_DEBUG_EVENT => {
                        threads += 1;
                        #[cfg(target_arch = "x86_64")]
                        if let Some(s) = sampler.as_mut() {
                            s.thread(event.dwThreadId, event.u.CreateThread.hThread);
                        }
                        #[cfg(target_arch = "x86_64")]
                        tools.thread(info.hProcess, event.dwThreadId, event.u.CreateThread.hThread, &mut |t: &str| log.line(&format!("{} {t}", stamp(start))));
                    }
                    #[cfg(target_arch = "x86_64")]
                    EXIT_THREAD_DEBUG_EVENT => {
                        if let Some(s) = sampler.as_mut() {
                            s.thread_gone(event.dwThreadId);
                        }
                        tools.thread_gone(event.dwThreadId);
                    }
                    OUTPUT_DEBUG_STRING_EVENT => {
                        let s = event.u.DebugString;
                        let text = debug_string(info.hProcess, s.lpDebugStringData.cast(), s.nDebugStringLength, s.fUnicode != 0);
                        for line in text.lines() {
                            log.line(&format!("{} debug {line}", stamp(start)));
                        }
                        #[cfg(target_arch = "x86_64")]
                        tools.debug_string(&text);
                    }
                    EXCEPTION_DEBUG_EVENT => {
                        let e = event.u.Exception;
                        let code = e.ExceptionRecord.ExceptionCode as u32;
                        let breakpoint = code == 0x8000_0003 || code == 0x4000_001F;
                        #[cfg(target_arch = "x86_64")]
                        let ours_hit = if let Some(h) = hook.as_mut() {
                            if code == 0x8000_0003 && h.owns(e.ExceptionRecord.ExceptionAddress as usize) {
                                let mut say = |t: &str| log.line(&format!("{} {t}", stamp(start)));
                                h.hit(info.hProcess, event.dwThreadId, e.ExceptionRecord.ExceptionAddress as usize, &modules, &mut say);
                                true
                            } else {
                                code == hook::SINGLE_STEP && h.stepped(info.hProcess, event.dwThreadId)
                            }
                        } else {
                            false
                        };
                        #[cfg(not(target_arch = "x86_64"))]
                        let ours_hit = false;
                        // A data breakpoint of DECANT_WATCH, or a thread name: handled by the tools.
                        #[cfg(target_arch = "x86_64")]
                        let ours_hit = ours_hit || {
                            let r = &e.ExceptionRecord;
                            let n = (r.NumberParameters as usize).min(r.ExceptionInformation.len());
                            tools.exception(info.hProcess, event.dwThreadId, code, r.ExceptionAddress as usize, &r.ExceptionInformation[..n], e.dwFirstChance != 0, &mut |t: &str| log.line(&format!("{} {t}", stamp(start))))
                        };
                        // AMD only instructions of a console title (CLZERO and others) on a CPU without them.
                        #[cfg(target_arch = "x86_64")]
                        let emulated = if !ours_hit && emulate_cpu && code == crate::cpu_emu::ILLEGAL_INSTRUCTION && e.dwFirstChance != 0 {
                            crate::cpu_emu::platform::emulate(info.hProcess, event.dwThreadId, &mut cpu_stats)
                        } else {
                            None
                        };
                        #[cfg(not(target_arch = "x86_64"))]
                        let emulated: Option<()> = None;
                        if let Some(done) = emulated {
                            #[cfg(target_arch = "x86_64")]
                            if done.first {
                                log.line(&format!("{} emulate {} at {}", stamp(start), done.insn.name(), locate(&modules, done.address)));
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            let () = done;
                        } else if ours_hit {
                            // The hook's own break or step: nothing to report.
                        } else if breakpoint && first_breakpoint {
                            // The loader's own breakpoint once the process is set up.
                            first_breakpoint = false;
                            log.line(&format!("{} loader ready", stamp(start)));
                        } else {
                            if code == 0xC000_0005 && e.dwFirstChance != 0 {
                                first_chance_av += 1;
                            }
                            let chance = if e.dwFirstChance != 0 { "first chance" } else { "UNHANDLED" };
                            let at = e.ExceptionRecord.ExceptionAddress as usize;
                            log.line(&format!("{} exception {code:#010x} {} at {at:#014x}, {} ({chance})", stamp(start), exception_name(code), locate(&modules, at)));
                            // An access violation names the access (0 read, 1 write, 8 execute), the
                            // address, and the registers once, on its first chance.
                            if code == 0xC000_0005 && e.dwFirstChance != 0 && e.ExceptionRecord.NumberParameters >= 2 {
                                let p = e.ExceptionRecord.ExceptionInformation;
                                log.line(&format!("{}   access {} at {:#x}", stamp(start), p[0], p[1]));
                                // What lies at the address and on the page below it: state, protection
                                // and region, so an unmapped page tells from a read only or guarded one.
                                #[cfg(target_arch = "x86_64")]
                                for at in [p[1], p[1].wrapping_sub(0x1000)] {
                                    log.line(&format!("{}   memory at {at:#x}: {}", stamp(start), crate::cpu_emu::platform::describe_memory(info.hProcess, at)));
                                }
                                #[cfg(target_arch = "x86_64")]
                                if let Some(regs) = crate::cpu_emu::platform::registers(event.dwThreadId) {
                                    log.line(&format!("{}   {regs}", stamp(start)));
                                }
                                #[cfg(target_arch = "x86_64")]
                                if let Some(calls) = title_calls(info.hProcess, event.dwThreadId, &modules, 256, 12) {
                                    log.line(&format!("{}   stack: {calls}", stamp(start)));
                                }
                            } else if e.dwFirstChance == 0 {
                                // Any other exception the title does not handle: where it came from.
                                #[cfg(target_arch = "x86_64")]
                                if let Some(calls) = title_calls(info.hProcess, event.dwThreadId, &modules, 512, 24) {
                                    log.line(&format!("{}   stack: {calls}", stamp(start)));
                                }
                                #[cfg(target_arch = "x86_64")]
                                for line in crate::cpu_emu::platform::registers_pointed(info.hProcess, event.dwThreadId) {
                                    log.line(&format!("{}   {line}", stamp(start)));
                                }
                            } else if code == 0xE06D_7363 {
                                // A C++ throw: where in the title, since a caught one can still end in a null answer.
                                #[cfg(target_arch = "x86_64")]
                                if let Some(calls) = title_calls(info.hProcess, event.dwThreadId, &modules, 256, 8) {
                                    log.line(&format!("{}   thread {} stack: {calls}", stamp(start), event.dwThreadId));
                                }
                            }
                            status = DBG_EXCEPTION_NOT_HANDLED;
                        }
                    }
                    RIP_EVENT => log.line(&format!("{} debugger error {}", stamp(start), event.u.RipInfo.dwError)),
                    EXIT_PROCESS_DEBUG_EVENT => {
                        let code = event.u.ExitProcess.dwExitCode;
                        let name = exception_name(code);
                        let name = if name.is_empty() { String::new() } else { format!(" {name}") };
                        log.line(&format!("{} exit {code} ({code:#010x}{name}), {threads} other threads started", stamp(start)));
                        outcome.exit_code = Some(code);
                        #[cfg(target_arch = "x86_64")]
                        if let Some(h) = hook.as_ref() {
                            for line in h.summary() {
                                log.line(&format!("{} {line}", stamp(start)));
                            }
                        }
                        #[cfg(target_arch = "x86_64")]
                        if let Some(s) = sampler.as_mut() {
                            s.write_report();
                        }
                        #[cfg(target_arch = "x86_64")]
                        if cpu_stats.total() > 0 {
                            log.line(&format!("{} {}", stamp(start), cpu_stats.summary()));
                        }
                        #[cfg(target_arch = "x86_64")]
                        tools.exit(&mut |t: &str| log.line(&format!("{} {t}", stamp(start))));
                    }
                    _ => {}
                }
                ContinueDebugEvent(event.dwProcessId, event.dwThreadId, status);
            }
            if outcome.exit_code.is_some() {
                break;
            }
        }
        // SAFETY: handles the call gave us.
        unsafe {
            CloseHandle(info.hThread);
            CloseHandle(info.hProcess);
        }
        Ok(outcome)
    }
}

/// Runs `exe` with `args` from `folder` (its application folder, where the shims are), adding
/// `extra_env` to the environment, and logs the whole run into `log`. With `timeout`, a title still
/// running then is stopped. For the length of the run the game drive `G:` is mapped to `folder`
/// (the title hardcodes it, see [`crate::drive`]) and removed after, also on error and cancel.
/// `DECANT_GAME_DRIVE=0` switches that off. A `G:` that is a real drive (or points elsewhere) is never
/// touched: the debugger redirects the title's file paths that start with `G:/` to `folder`
/// instead (`DECANT_GAME_DRIVE_MODE=redirect` forces this, for tests).
pub fn run(exe: &Path, args: &[String], folder: &Path, extra_env: &[(String, String)], timeout: Option<Duration>, log: &mut dyn Sink) -> std::io::Result<Outcome> {
    run_cancellable(exe, args, folder, extra_env, timeout, &AtomicBool::new(false), log)
}

/// Like [`run`], and a title still running when another thread sets `cancel` is stopped (the
/// outcome has `cancelled` set). The flag is read about every 100 ms.
#[cfg(windows)]
pub fn run_cancellable(exe: &Path, args: &[String], folder: &Path, extra_env: &[(String, String)], timeout: Option<Duration>, cancel: &AtomicBool, log: &mut dyn Sink) -> std::io::Result<Outcome> {
    let redirect = redirect_enabled(std::env::var(REDIRECT_ENV).ok().as_deref());
    run_with_redirect(exe, args, folder, extra_env, timeout, cancel, redirect, log)
}

/// Like [`run_cancellable`], with the redirect of `xgameruntime.dll` to the copy in `folder` (the
/// interposer) set by `redirect` instead of by [`REDIRECT_ENV`]. A title asks for the runtime by
/// name with `LOAD_LIBRARY_SEARCH_SYSTEM32`, which skips its own folder; the debugger answers that
/// call with the full path of the interposer. Nothing happens without `xgameruntime.dll` in `folder`.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
pub fn run_with_redirect(exe: &Path, args: &[String], folder: &Path, extra_env: &[(String, String)], timeout: Option<Duration>, cancel: &AtomicBool, redirect: bool, log: &mut dyn Sink) -> std::io::Result<Outcome> {
    let mut drive = crate::drive::GameDrive::acquire(folder, log);
    run_with_drive(exe, args, folder, extra_env, timeout, cancel, redirect, &mut drive, log)
}

/// Like [`run_with_redirect`] with the game drive already chosen (see [`crate::drive::GameDrive`]):
/// a drive in redirect mode sends the title's file paths under its letter to `folder` by the
/// debugger, a mapped one is removed at the end. The persistent local storage drive is mapped for
/// the run as well (see [`crate::pls_drive`]).
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
pub fn run_with_drive(exe: &Path, args: &[String], folder: &Path, extra_env: &[(String, String)], timeout: Option<Duration>, cancel: &AtomicBool, redirect: bool, drive: &mut crate::drive::GameDrive, log: &mut dyn Sink) -> std::io::Result<Outcome> {
    use crate::relaunch;
    let mut pls = crate::pls_drive::PlsDrive::acquire(exe.parent().unwrap_or(folder), log);
    // A title that asks for a relaunch (`XLaunchNewGame`) is started again with its new arguments,
    // within the same timeout (see `relaunch`).
    let file = folder.join(relaunch::FILE_NAME);
    let mut env: Vec<(String, String)> = extra_env.iter().cloned().chain(pls.env().iter().cloned()).collect();
    env.push((relaunch::FILE_ENV.to_owned(), file.to_string_lossy().into_owned()));
    let (mut exe, mut args) = (exe.to_path_buf(), args.to_vec());
    let deadline = timeout.map(|t| Instant::now() + t);
    let mut relaunches = 0;
    let outcome = loop {
        let _ = std::fs::remove_file(&file);
        let left = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        let outcome = platform::run(&exe, &args, folder, &env, left, cancel, redirect, drive.redirect(), log);
        let Ok(done) = &outcome else { break outcome };
        let Some(request) = relaunch::requested(done, &file) else { break outcome };
        if relaunches == relaunch::MAX || left.is_some_and(|l| l.is_zero()) {
            log.line(&format!("relaunch asked ({} {:?}) but refused: {relaunches} relaunches already or no time left", request.exe, request.args));
            break outcome;
        }
        relaunches += 1;
        exe = relaunch::executable(&request.exe, &exe, folder);
        args = request.args;
        log.line(&format!("relaunch {relaunches}: the title asked to be started again (XLaunchNewGame): {} {:?}", exe.display(), args));
    };
    let _ = std::fs::remove_file(&file);
    pls.release(log);
    drive.release(log);
    outcome
}

#[cfg(not(windows))]
pub fn run_cancellable(_: &Path, _: &[String], _: &Path, _: &[(String, String)], _: Option<Duration>, _: &AtomicBool, _: &mut dyn Sink) -> std::io::Result<Outcome> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "decant-launch runs titles on Windows; on Linux use launch_prep (crate decant-linux)"))
}

#[cfg(not(windows))]
#[allow(clippy::too_many_arguments)]
pub fn run_with_redirect(_: &Path, _: &[String], _: &Path, _: &[(String, String)], _: Option<Duration>, _: &AtomicBool, _: bool, _: &mut dyn Sink) -> std::io::Result<Outcome> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "decant-launch runs titles on Windows; on Linux use launch_prep (crate decant-linux)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_lines_are_quoted_as_the_crt_reads_them() {
        let exe = Path::new(r"C:\Games\My Title\Title.exe");
        let args = ["-plain".to_owned(), "two words".to_owned(), r#"say "hi""#.to_owned(), r"ends\".to_owned(), String::new()];
        assert_eq!(command_line(exe, &args), r#""C:\Games\My Title\Title.exe" -plain "two words" "say \"hi\"" ends\ """#);
    }

    #[test]
    fn the_redirect_is_on_unless_switched_off() {
        assert!(redirect_enabled(None));
        assert!(redirect_enabled(Some("1")));
        assert!(redirect_enabled(Some("")));
        for off in ["0", "off", "FALSE", " No "] {
            assert!(!redirect_enabled(Some(off)), "{off}");
        }
    }

    #[test]
    fn only_requests_for_the_runtime_are_redirected() {
        let ours = Path::new(r"D:\Title\xgameruntime.dll");
        assert!(is_runtime_request("xgameruntime.dll", ours));
        assert!(is_runtime_request("XGameRuntime.DLL", ours));
        assert!(is_runtime_request(r"C:\Windows\system32\xgameruntime.dll", ours));
        assert!(is_runtime_request("C:/Windows/System32/xgameruntime.dll", ours));
        assert!(!is_runtime_request(r"d:\title\XGAMERUNTIME.dll", ours));
        assert!(!is_runtime_request("xgameruntime.dll.bak", ours));
        assert!(!is_runtime_request("kernel32.dll", ours));
        assert!(!is_runtime_request("", ours));
    }

    /// The keep list for these entries (see `keep_entry`), on the drive `G`.
    fn keep(entries: &[&str]) -> Vec<String> {
        entries.iter().filter_map(|e| keep_entry(e, 'G')).collect()
    }

    #[test]
    fn game_drive_paths_are_rewritten_to_the_folder() {
        let folder = r"D:\title_overlay";
        let t = keep(&[]);
        assert_eq!(redirect_game_path("G:/package.dat", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\package.dat"));
        assert_eq!(redirect_game_path(r"g:\x\y", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\x\y"));
        assert_eq!(redirect_game_path("G:/a/b\\c.dat", 'g', r"D:\title_overlay\", &t).as_deref(), Some(r"D:\title_overlay\a\b\c.dat"));
        assert_eq!(redirect_game_path("G:/", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\"));
        assert_eq!(redirect_game_path(r"G:\", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\"));
        assert_eq!(redirect_game_path("G://x", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\x"));
        assert_eq!(redirect_game_path(r"G:\data\levels\a.dat", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\data\levels\a.dat"));
        // The root with no names at all still maps.
        assert_eq!(redirect_game_path("G:/", 'G', folder, &[]).as_deref(), Some(r"D:\title_overlay\"));
    }

    #[test]
    fn extended_device_and_nt_forms_of_a_game_drive_path_are_redirected_like_the_plain_one() {
        let folder = r"D:\title_overlay";
        let t = keep(&[]);
        let plain = Some(r"D:\title_overlay\pack\core\a.ttpf".to_owned());
        for p in [r"G:\pack\core\a.ttpf", r"\\?\G:\pack\core\a.ttpf", r"\\.\g:\pack\core\a.ttpf", r"\??\G:\pack\core\a.ttpf", "//?/G:/pack/core/a.ttpf", "/??/G:/pack/core/a.ttpf", r"\\?\G:/pack\core/a.ttpf"] {
            assert_eq!(redirect_game_path(p, 'G', folder, &t), plain, "{p}");
        }
        assert_eq!(redirect_game_path(r"\\?\G:\", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\"));
        // Inside the title folder in any form, another drive and a UNC path stay as they are.
        assert_eq!(redirect_game_path(r"\??\G:\Work\overlay\x", 'G', r"G:\Work\overlay", &t), None);
        assert_eq!(redirect_game_path(r"\\?\H:\pack\a", 'G', folder, &t), None);
        assert_eq!(redirect_game_path(r"\\?\UNC\server\share\pack", 'G', folder, &t), None);
    }

    #[test]
    fn the_first_component_is_matched_ignoring_case() {
        let folder = r"D:\title_overlay";
        let t = keep(&[]);
        assert_eq!(redirect_game_path("G:/COMMON.dat", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\COMMON.dat"));
        assert_eq!(redirect_game_path(r"G:\dAtA/x\y.dat", 'G', folder, &t).as_deref(), Some(r"D:\title_overlay\dAtA\x\y.dat"));
    }

    #[test]
    fn every_path_of_the_drive_is_redirected_except_the_keep_list() {
        let folder = r"D:\title_overlay";
        // The title's own paths, whatever their first folder (`G:\pack`, `G://settings/`, one the folder lacks).
        let none = keep(&[]);
        for (p, want) in [
            (r"G:\pack\core\a.ttpf", r"D:\title_overlay\pack\core\a.ttpf"),
            ("G://settings/x.xml", r"D:\title_overlay\settings\x.xml"),
            ("G:/Games/x.txt", r"D:\title_overlay\Games\x.txt"),
            (r"G:\package.dat.bak", r"D:\title_overlay\package.dat.bak"),
        ] {
            assert_eq!(redirect_game_path(p, 'G', folder, &none).as_deref(), Some(want), "{p}");
        }
        // A scratch folder on the real drive named in the environment stays where it is, in any form of
        // the path: mixed slashes, any case, extended prefix.
        let scratch = game_drive_keep('G', &[(KEEP_ENV.to_owned(), r"G:\Tools\Scratch".to_owned())]);
        for p in [r"G:\Tools\Scratch\work\x.bin", r"G:\Tools\Scratch\work/5ec439b66277e3af.bin", r"G:\TOOLS\scratch", r"\\?\G:\Tools\Scratch\a"] {
            assert_eq!(classify_game_path(p, 'G', folder, &scratch), Verdict::Kept("on the keep list of the game drive"), "{p}");
        }
        // A sibling that starts the same is not the kept folder, and nothing is kept without a list.
        assert!(redirect_game_path(r"G:\Tools\Scratch2\a", 'G', folder, &scratch).is_some());
        assert!(redirect_game_path(r"G:\Tools\Other", 'G', folder, &scratch).is_some());
        assert!(game_drive_keep('G', &[]).is_empty());
    }

    #[test]
    fn the_keep_list_comes_from_the_environment_and_never_covers_the_whole_drive() {
        let env: Vec<(String, String)> = [
            ("DECANT_GAME_DRIVE_KEEP", r"G:\Mine\A;/other/b/; H:\elsewhere ; G:\ ; ;\\?\G:\Z"),
            ("DECANT_D3D12X_DXIL_OVERRIDE", r"G:\Content\Cache_1"),
            ("DECANT_D3D12X_PROBE_DUMP", r"C:\dump"),
            ("DECANT_KEYS", "100:A,125:A"),
            ("PATH", r"G:\tools"),
        ]
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect();
        let got = game_drive_keep('G', &env);
        let mut want: Vec<String> = ["content\\cache_1", "mine\\a", "other\\b", "z"].iter().map(|s| (*s).to_owned()).collect();
        want.sort();
        assert_eq!(got, want);
        let folder = r"D:\title_overlay";
        assert_eq!(classify_game_path(r"G:\CONTENT\cache_1\a.bin", 'G', folder, &got), Verdict::Kept("on the keep list of the game drive"));
        assert!(redirect_game_path(r"G:\Content\pack\a", 'G', folder, &got).is_some());
    }

    #[test]
    fn other_paths_are_left_alone() {
        let folder = r"D:\title_overlay";
        let t = keep(&[]);
        for p in ["", "G", "G:", "G:package.dat", "H:/package.dat", r"C:\G:\x", "xG:/a", r"\\?\UNC\G:\a", r"\\?\G:", "package.dat", "/G:/a"] {
            assert_eq!(redirect_game_path(p, 'G', folder, &t), None, "{p}");
        }
        assert_eq!(redirect_game_path("H:/a", 'H', folder, &t).as_deref(), Some(r"D:\title_overlay\a"));
        // The title folder itself on the game drive (G: is a real drive): its own paths stay as they are.
        let on_g = r"G:\Work\boot\overlay";
        let t = keep(&[]);
        assert_eq!(redirect_game_path(r"G:\Work\boot\overlay\decant_storage_abi.txt", 'G', on_g, &t), None);
        assert_eq!(redirect_game_path("g:/work/BOOT/overlay/a.dat", 'G', on_g, &t), None);
        assert_eq!(redirect_game_path(r"G:\Work\boot\overlay", 'G', on_g, &t), None);
        assert_eq!(redirect_game_path(r"G:\package.dat", 'G', on_g, &t).as_deref(), Some(r"G:\Work\boot\overlay\package.dat"));
        // Not in the overlay (another folder that starts the same): redirected like any path of the drive.
        assert_eq!(redirect_game_path(r"G:\Work\boot\overlay2\x", 'G', on_g, &t).as_deref(), Some(r"G:\Work\boot\overlay\Work\boot\overlay2\x"));
    }

    #[test]
    fn paths_already_in_an_overlay_on_the_game_drive_are_left_alone() {
        let folder = r"G:\Work\overlay_hote";
        let t = keep(&[]);
        for p in [r"G:\Work\overlay_hote\decant_storage_abi.txt", "g:/work/Overlay_Hote/a/b.dat", r"G:\Work\overlay_hote", r"G:\Work\overlay_hote\"] {
            assert_eq!(redirect_game_path(p, 'G', folder, &t), None, "{p}");
        }
        assert_eq!(redirect_game_path("G:/package.dat", 'G', folder, &t).as_deref(), Some(r"G:\Work\overlay_hote\package.dat"));
    }

    #[test]
    fn long_results_get_the_extended_prefix() {
        let folder = r"D:\title_overlay";
        let t = keep(&[]);
        let short = format!("G:/{}", "a".repeat(200));
        assert!(!redirect_game_path(&short, 'G', folder, &t).unwrap().starts_with(r"\\?\"));
        let long = format!("G:/{}/{}", "b".repeat(150), "c".repeat(150));
        let out = redirect_game_path(&long, 'G', folder, &t).unwrap();
        assert!(out.starts_with(r"\\?\D:\title_overlay\bbb") && out.ends_with("ccc"), "{out}");
        let unc = redirect_game_path(&long, 'G', r"\\server\share\t", &t).unwrap();
        assert!(unc.starts_with(r"\\server\share\t\"), "{unc}");
    }

    #[test]
    fn a_caller_is_named_by_the_nearest_module_below_it() {
        let modules = vec![(0x1000, "a.exe".to_owned()), (0x9000, "b.dll".to_owned())];
        assert_eq!(locate(&modules, 0x9010), "b.dll+0x10");
        assert_eq!(locate(&modules, 0x2000), "a.exe+0x1000");
        assert_eq!(locate(&modules, 0x10), "0x10");
    }

    #[cfg(windows)]
    #[test]
    fn a_process_run_under_the_debugger_logs_its_dlls_and_exit_code() {
        let system = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let cmd = Path::new(&system).join("System32").join("cmd.exe");
        let folder = std::env::temp_dir();
        let mut log: Vec<String> = Vec::new();
        let outcome = run(&cmd, &["/C".into(), "exit 7".into()], &folder, &[("DECANT_LOG".into(), "1".into())], Some(Duration::from_secs(30)), &mut log).unwrap();
        assert_eq!(outcome.exit_code, Some(7));
        assert!(!outcome.timed_out);
        assert!(outcome.dlls.iter().any(|p| p.to_string_lossy().to_lowercase().ends_with("kernel32.dll")), "{log:#?}");
        assert!(log.iter().any(|l| l.contains("loader ready")));
        // After the exit line only the file-call counts come (one line per function used) and, when this
        // run mapped the drive, the line that says it is removed.
        let exit = log.iter().rposition(|l| l.contains("exit 7")).expect("an exit line");
        assert!(log[exit + 1..].iter().all(|l| l.contains("file calls:") || l.to_lowercase().contains("drive")), "{log:#?}");
    }

    #[cfg(windows)]
    #[test]
    fn the_title_runs_without_the_debug_heap_unless_asked() {
        let system = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let cmd = Path::new(&system).join("System32").join("cmd.exe");
        let folder = std::env::temp_dir();
        let check = || vec!["/C".to_string(), "if \"%_NO_DEBUG_HEAP%\"==\"1\" (exit 3) else (exit 4)".to_string()];
        let mut log: Vec<String> = Vec::new();
        let off = run(&cmd, &check(), &folder, &[], Some(Duration::from_secs(30)), &mut log).unwrap();
        assert_eq!(off.exit_code, Some(3), "{log:#?}");
        let kept = run(&cmd, &check(), &folder, &[("_NO_DEBUG_HEAP".into(), "0".into())], Some(Duration::from_secs(30)), &mut log).unwrap();
        assert_eq!(kept.exit_code, Some(4), "an explicit value stays");
    }

    #[cfg(windows)]
    #[test]
    fn a_raised_cancel_flag_stops_a_running_process() {
        let system = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let cmd = Path::new(&system).join("System32").join("cmd.exe");
        let folder = std::env::temp_dir();
        let cancel = AtomicBool::new(false);
        let mut log: Vec<String> = Vec::new();
        let started = Instant::now();
        let outcome = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(500));
                cancel.store(true, Ordering::Relaxed);
            });
            run_cancellable(&cmd, &["/C".into(), "ping -n 60 127.0.0.1 >nul".into()], &folder, &[], None, &cancel, &mut log).unwrap()
        });
        assert!(outcome.cancelled && !outcome.timed_out, "{log:#?}");
        assert!(outcome.exit_code.is_some());
        assert!(started.elapsed() < Duration::from_secs(20));
        assert!(log.iter().any(|l| l.contains("cancelled")));
    }
}
