//! A picture of every thread of the title at one moment, for `DECANT_STACKS` and `DECANT_WAITS`: all
//! threads are suspended, each one's registers are read and its stack walked, then all are resumed.
//! A thread that sits in a wait is named with the wait functions on top of its stack and the
//! objects it waits for (kind, name, state, and who created it when one of our DLLs said so).

/// The functions of `ntdll.dll` and `win32u.dll` a waiting thread sits in, and how to read what it
/// waits for from its registers at the system call (the first argument is in r10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitArgs {
    /// One handle, first argument.
    Handle,
    /// A count in the first argument and an array of handles in the second.
    Handles,
    /// An address, first argument (SRW locks, condition variables, `WaitOnAddress`, critical sections).
    Address,
    /// Nothing worth reading (a sleep, the thread pool's idle wait, a message wait).
    Nothing,
}

/// What kind of wait the system call `name` is, `None` when it is no wait.
pub fn wait_kind(name: &str) -> Option<WaitArgs> {
    let name = name.strip_prefix("Zw").map_or(name.to_owned(), |rest| format!("Nt{rest}"));
    Some(match name.as_str() {
        "NtWaitForSingleObject" | "NtRemoveIoCompletion" | "NtRemoveIoCompletionEx" | "NtSignalAndWaitForSingleObject" | "NtWaitForKeyedEvent" | "NtReadFile" | "NtDeviceIoControlFile" | "NtFsControlFile" => WaitArgs::Handle,
        "NtWaitForMultipleObjects" | "NtWaitForMultipleObjects32" | "NtUserMsgWaitForMultipleObjectsEx" => WaitArgs::Handles,
        "NtWaitForAlertByThreadId" => WaitArgs::Address,
        "NtDelayExecution" | "NtWaitForWorkViaWorkerFactory" | "NtUserGetMessage" | "NtUserWaitMessage" | "NtWaitForDebugEvent" | "NtYieldExecution" => WaitArgs::Nothing,
        _ => return None,
    })
}

/// The export name of a frame text `module+rva (module!Name+0x..)`, if it has one.
pub fn export_of(frame: &str) -> Option<&str> {
    let inner = frame.split_once(" (")?.1.strip_suffix(')')?;
    let name = inner.split_once('!')?.1;
    Some(name.rsplit_once('+').map_or(name, |(n, _)| n))
}

/// The leading frames that are exports of system modules, joined: `NtWaitForSingleObject <
/// WaitForSingleObjectEx`. This names the wait the title asked for.
pub fn wait_chain(frames: &[String]) -> String {
    frames.iter().take_while(|f| export_of(f).is_some()).filter_map(|f| export_of(f)).take(4).collect::<Vec<_>>().join(" < ")
}

#[cfg(all(windows, target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub mod platform {
    use super::*;
    use crate::tools::Threads;
    use crate::tools::unwind::{self, Memory, Modules, platform::Process};
    use std::collections::HashMap;
    use std::ffi::c_void;
    use windows_sys::Win32::Foundation::{CloseHandle, DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64, GetThreadContext};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessId, GetThreadId, ResumeThread, SuspendThread};

    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn NtQueryObject(handle: HANDLE, class: u32, info: *mut c_void, length: u32, returned: *mut u32) -> i32;
        fn NtQueryEvent(handle: HANDLE, class: u32, info: *mut c_void, length: u32, returned: *mut u32) -> i32;
        fn NtQuerySemaphore(handle: HANDLE, class: u32, info: *mut c_void, length: u32, returned: *mut u32) -> i32;
        fn NtQueryMutant(handle: HANDLE, class: u32, info: *mut c_void, length: u32, returned: *mut u32) -> i32;
    }

    #[repr(C, align(16))]
    struct Aligned(CONTEXT);

    /// One thread at the moment of the picture.
    pub struct ThreadState {
        pub tid: u32,
        pub label: String,
        pub frames: Vec<String>,
        /// The wait it sits in: the wait functions and what it waits for.
        pub blocked: Option<String>,
        /// What identifies this wait: same key in two pictures, same wait.
        pub key: String,
    }

    /// Who created a handle, as our DLLs reported it (`[decant-handle]` debug lines).
    pub type Creators = HashMap<usize, String>;

    /// Kind, name and state of a handle of the title.
    // process is an opaque handle the OS checks, never dereferenced here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn describe_handle(process: HANDLE, handle: usize, creators: &Creators) -> String {
        let mut dup: HANDLE = std::ptr::null_mut();
        // SAFETY: duplicates the title's handle into this process to look at it.
        let ok = unsafe { DuplicateHandle(process, handle as HANDLE, GetCurrentProcess(), &mut dup, 0, 0, DUPLICATE_SAME_ACCESS) } != 0;
        let mut out = format!("{handle:#x}");
        if ok {
            let kind = query_string(dup, 2).unwrap_or_else(|| "?".into());
            out += &format!(" {kind}");
            // Only these kinds: a name query on a file can block.
            if matches!(kind.as_str(), "Event" | "Semaphore" | "Mutant" | "Section" | "Timer" | "IoCompletion")
                && let Some(name) = query_string(dup, 1).filter(|n| !n.is_empty())
            {
                out += &format!(" \"{name}\"");
            }
            out += &state(dup, &kind);
            // SAFETY: our duplicate.
            unsafe { CloseHandle(dup) };
        }
        if let Some(who) = creators.get(&handle) {
            out += &format!(", created by {who}");
        }
        out
    }

    /// A UNICODE_STRING answer of NtQueryObject: class 1 the name, class 2 the type name.
    fn query_string(handle: HANDLE, class: u32) -> Option<String> {
        let mut buf = vec![0u64; 512];
        let mut got = 0u32;
        // SAFETY: the buffer is as long as the length passed.
        let status = unsafe { NtQueryObject(handle, class, buf.as_mut_ptr().cast(), (buf.len() * 8) as u32, &mut got) };
        if status < 0 {
            return None;
        }
        // UNICODE_STRING { u16 Length, u16 MaximumLength, u32 pad, *u16 Buffer } at the start.
        let length = (buf[0] & 0xFFFF) as usize / 2;
        let pointer = buf[1] as *const u16;
        if pointer.is_null() || length == 0 {
            return Some(String::new());
        }
        // SAFETY: the string lies inside `buf`, which the call filled.
        Some(String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(pointer, length) }))
    }

    fn state(handle: HANDLE, kind: &str) -> String {
        let mut info = [0i32; 4];
        let mut got = 0u32;
        // SAFETY: each call fills at most `size` bytes of `info`. None of them changes the object.
        unsafe {
            match kind {
                "Event" if NtQueryEvent(handle, 0, info.as_mut_ptr().cast(), 8, &mut got) >= 0 => {
                    format!(" ({}, {})", if info[0] == 0 { "manual reset" } else { "auto reset" }, if info[1] != 0 { "set" } else { "not set" })
                }
                "Semaphore" if NtQuerySemaphore(handle, 0, info.as_mut_ptr().cast(), 8, &mut got) >= 0 => format!(" (count {} of {})", info[0], info[1]),
                "Mutant" => {
                    let mut owner = [0u64; 2];
                    if NtQueryMutant(handle, 0, info.as_mut_ptr().cast(), 12, &mut got) < 0 {
                        return String::new();
                    }
                    let held = if info[0] <= 0 { "held" } else { "free" };
                    if NtQueryMutant(handle, 1, owner.as_mut_ptr().cast(), 16, &mut got) >= 0 && owner[1] != 0 {
                        format!(" ({held}, owner thread {})", owner[1])
                    } else {
                        format!(" ({held})")
                    }
                }
                "Thread" => format!(" {}", GetThreadId(handle)),
                "Process" => format!(" {}", GetProcessId(handle)),
                _ => String::new(),
            }
        }
    }

    /// The kind of a handle of the title, `None` when it is none.
    fn handle_kind(process: HANDLE, handle: usize) -> Option<String> {
        let mut dup: HANDLE = std::ptr::null_mut();
        // SAFETY: duplicates the title's handle into this process to look at it.
        if unsafe { DuplicateHandle(process, handle as HANDLE, GetCurrentProcess(), &mut dup, 0, 0, DUPLICATE_SAME_ACCESS) } == 0 {
            return None;
        }
        let kind = query_string(dup, 2);
        // SAFETY: our duplicate.
        unsafe { CloseHandle(dup) };
        kind
    }

    /// What a thread in a handle wait waits for. The handles were passed in volatile registers,
    /// which the kernel does not keep for a thread inside a system call, so they are looked for
    /// where the calling wait function must keep them to retry after an alert: in the non volatile
    /// registers and on the stack up to the first frame outside the system modules. Every value
    /// there that is a waitable handle of the title is named.
    fn what_it_waits_for(mem: &Process, process: HANDLE, c: &CONTEXT, kind: WaitArgs, depth_bytes: usize, creators: &Creators) -> String {
        if !matches!(kind, WaitArgs::Handle | WaitArgs::Handles) {
            return String::new();
        }
        let mut values: Vec<u64> = vec![c.Rbx, c.Rbp, c.Rsi, c.Rdi, c.R12, c.R13, c.R14, c.R15];
        let words = depth_bytes.clamp(64, 0x400) / 8;
        values.extend((0..words).filter_map(|i| mem.u64_at(c.Rsp as usize + i * 8)));
        let mut seen = Vec::new();
        let mut found = Vec::new();
        for v in values {
            let h = v as usize;
            if h == 0 || h & 3 != 0 || h >= 1 << 24 || seen.contains(&h) {
                continue;
            }
            seen.push(h);
            let Some(kind) = handle_kind(process, h) else { continue };
            if matches!(kind.as_str(), "Event" | "Semaphore" | "Mutant" | "Thread" | "Process" | "Timer" | "IoCompletion" | "WaitCompletionPacket" | "Job" | "File") {
                found.push(describe_handle(process, h, creators));
            }
        }
        match found.len() {
            0 => "no waitable handle found near the wait".into(),
            1 => format!("on {}", found[0]),
            _ => format!("on one or more of: {}", found.join("; ")),
        }
    }

    /// Suspends every thread, reads and walks each one, resumes them all.
    pub fn take(mem: &Process, modules: &Modules, threads: &Threads, frames: usize, creators: &Creators) -> Vec<ThreadState> {
        let all: Vec<(u32, HANDLE)> = threads.iter().map(|(&tid, t)| (tid, t.handle)).collect();
        // SAFETY: debuggee thread handles with THREAD_SUSPEND_RESUME, each resumed below.
        let suspended: Vec<bool> = all.iter().map(|(_, h)| unsafe { SuspendThread(*h) } != u32::MAX).collect();
        let mut out = Vec::new();
        for ((tid, handle), ok) in all.iter().zip(&suspended) {
            if !ok {
                continue;
            }
            // SAFETY: a plain structure the call fills.
            let mut c: Aligned = unsafe { std::mem::zeroed() };
            c.0.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
            // SAFETY: a debuggee thread handle with THREAD_GET_CONTEXT, the thread is suspended.
            if unsafe { GetThreadContext(*handle, &mut c.0) } == 0 {
                continue;
            }
            let walked_regs = unwind::walk_frames(mem, modules, unwind::platform::regs(&c.0), frames);
            let walked: Vec<usize> = walked_regs.iter().map(|r| r.rip as usize).collect();
            let sps: Vec<usize> = walked_regs.iter().map(|r| r.gpr[4] as usize).collect();
            let described = unwind::describe(mem, modules, &walked);
            let top = described.first().and_then(|f| export_of(f)).and_then(wait_kind);
            let blocked = top.map(|kind| {
                let chain = wait_chain(&described);
                // The stack of the system frames: up to the stack pointer of the first other frame.
                let system = described.iter().take_while(|f| export_of(f).is_some()).count();
                let depth = sps.get(system).map_or(0, |&sp| sp.saturating_sub(c.0.Rsp as usize));
                let on = what_it_waits_for(mem, mem.0, &c.0, kind, depth, creators);
                if on.is_empty() { chain } else { format!("{chain}: {on}") }
            });
            let key = format!("{}|{}", blocked.as_deref().unwrap_or("running"), walked.iter().map(|f| format!("{f:x}")).collect::<Vec<_>>().join(","));
            out.push(ThreadState { tid: *tid, label: threads.label(*tid), frames: described, blocked, key });
        }
        for ((_, handle), ok) in all.iter().zip(&suspended) {
            if *ok {
                // SAFETY: suspended above.
                unsafe { ResumeThread(*handle) };
            }
        }
        out
    }

    /// The picture as text, one block per thread.
    pub fn format(states: &[ThreadState]) -> String {
        let mut out = String::new();
        for s in states {
            out += &format!("{}: {}\n", s.label, s.blocked.as_deref().map_or_else(|| "running or ready".to_owned(), |b| format!("waits in {b}")));
            for f in &s.frames {
                out += &format!("    {f}\n");
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waits_are_known_by_their_system_call() {
        assert_eq!(wait_kind("NtWaitForSingleObject"), Some(WaitArgs::Handle));
        assert_eq!(wait_kind("ZwWaitForMultipleObjects"), Some(WaitArgs::Handles));
        assert_eq!(wait_kind("NtWaitForAlertByThreadId"), Some(WaitArgs::Address));
        assert_eq!(wait_kind("NtDelayExecution"), Some(WaitArgs::Nothing));
        assert_eq!(wait_kind("NtCreateFile"), None);
    }

    #[test]
    fn the_chain_is_the_leading_exports() {
        let frames: Vec<String> = [
            "ntdll.dll+0x9d5d4 (ntdll.dll!NtWaitForSingleObject+0x14)",
            "KERNELBASE.dll+0x3a4e (KERNELBASE.dll!WaitForSingleObjectEx+0x8e)",
            "title.exe+0x1234",
            "kernel32.dll+0x10 (kernel32.dll!BaseThreadInitThunk+0x10)",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(export_of(&frames[0]), Some("NtWaitForSingleObject"));
        assert_eq!(export_of("title.exe+0x1234"), None);
        assert_eq!(wait_chain(&frames), "NtWaitForSingleObject < WaitForSingleObjectEx");
    }
}
