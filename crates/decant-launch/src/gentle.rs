//! Gentle mode: the title runs at a lower priority so the desktop of the PC it runs on stays usable.
//! `DECANT_GENTLE=1` (or `decant-launch ... --gentle`) lowers, for every process the launcher starts for the
//! title (relaunches included), right after it is created:
//!
//! 1. the CPU priority class, to below normal,
//! 2. the GPU scheduling priority class (`D3DKMTSetProcessSchedulingPriorityClass`), to below normal.
//!    The kernel refuses it (`STATUS_INVALID_PARAMETER`) for a process that has not used the GPU yet,
//!    so the launcher tries again twice a second until it takes (the title opens its adapter while it
//!    boots) and logs a second line then.
//!
//! The memory priority is not lowered: a low memory priority makes the system trim the title's pages
//! first, and nothing shows that the title's own loading does not thrash then. One log line says what
//! was applied, and a refused step never fails the launch.

use std::time::Duration;

/// Environment variable that switches the mode on.
pub const ENV: &str = "DECANT_GENTLE";

/// Whether the mode is on for this value of [`ENV`] (unset means off).
pub fn enabled(value: Option<&str>) -> bool {
    value.is_some_and(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "off" | "false" | "no"))
}

/// `D3DKMT_SCHEDULINGPRIORITYCLASS_BELOW_NORMAL` of d3dkmthk.h (IDLE 0, BELOW_NORMAL 1, NORMAL 2,
/// ABOVE_NORMAL 3, HIGH 4, REALTIME 5).
pub const GPU_BELOW_NORMAL: i32 = 1;

/// How often the GPU step is tried again until it takes.
pub const RETRY: Duration = Duration::from_millis(500);

/// The log line for the outcome of the two steps, `Err` carrying the reason.
pub fn describe(cpu: &Result<(), String>, gpu: &Result<(), String>) -> String {
    let part = |name: &str, r: &Result<(), String>| match r {
        Ok(()) => format!("{name} below normal"),
        Err(why) => format!("{name} not applied ({why})"),
    };
    format!("DECANT_GENTLE: {}, {}, memory priority left alone", part("CPU priority", cpu), part("GPU scheduling priority", gpu))
}

#[cfg(windows)]
#[allow(unsafe_code)]
#[allow(clippy::not_unsafe_ptr_arg_deref)] // a process HANDLE is an opaque value the kernel checks, never dereferenced here
mod platform {
    use super::*;
    use std::time::Instant;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
    use windows_sys::Win32::System::Threading::{BELOW_NORMAL_PRIORITY_CLASS, SetPriorityClass};

    type SetClass = unsafe extern "system" fn(HANDLE, i32) -> i32;
    type GetClass = unsafe extern "system" fn(HANDLE, *mut i32) -> i32;

    fn gdi32(name: &[u8]) -> Option<unsafe extern "system" fn() -> isize> {
        // SAFETY: NUL-terminated names; gdi32.dll is a system DLL.
        unsafe {
            let module = LoadLibraryA(c"gdi32.dll".as_ptr().cast());
            if module.is_null() {
                return None;
            }
            GetProcAddress(module, name.as_ptr())
        }
    }

    /// `D3DKMTSetProcessSchedulingPriorityClass(process, class)` of d3dkmthk.h. The handle needs
    /// `PROCESS_SET_INFORMATION`.
    pub fn set_gpu_class(process: HANDLE, class: i32) -> Result<(), String> {
        let f = gdi32(b"D3DKMTSetProcessSchedulingPriorityClass\0").ok_or("not in gdi32.dll")?;
        // SAFETY: the export has this signature (NTSTATUS f(HANDLE, enum)).
        let status = unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, SetClass>(f)(process, class) };
        if status < 0 { Err(format!("status {:#x}", status as u32)) } else { Ok(()) }
    }

    /// `D3DKMTGetProcessSchedulingPriorityClass(process, &class)`. The handle needs `PROCESS_QUERY_INFORMATION` and `PROCESS_SET_INFORMATION` (access denied without the second).
    pub fn gpu_class(process: HANDLE) -> Result<i32, String> {
        let f = gdi32(b"D3DKMTGetProcessSchedulingPriorityClass\0").ok_or("not in gdi32.dll")?;
        let mut class = 0i32;
        // SAFETY: the export has this signature (NTSTATUS f(HANDLE, enum*)).
        let status = unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, GetClass>(f)(process, &raw mut class) };
        if status < 0 { Err(format!("status {:#x}", status as u32)) } else { Ok(class) }
    }

    /// The gentle state of one started process.
    pub struct Gentle {
        gpu_done: bool,
        last_try: Instant,
        started: Instant,
    }

    impl Gentle {
        /// Lowers the priorities of `process` and returns the state and the log line.
        pub fn apply(process: HANDLE) -> (Self, String) {
            // SAFETY: the handle is the creator's, with full access.
            let cpu = if unsafe { SetPriorityClass(process, BELOW_NORMAL_PRIORITY_CLASS) } != 0 { Ok(()) } else { Err(std::io::Error::last_os_error().to_string()) };
            let gpu = set_gpu_class(process, GPU_BELOW_NORMAL);
            let now = Instant::now();
            (Self { gpu_done: gpu.is_ok(), last_try: now, started: now }, describe(&cpu, &gpu))
        }

        /// Tries the GPU step again when it is due and not done. Returns a log line when it took.
        pub fn poll(&mut self, process: HANDLE) -> Option<String> {
            if self.gpu_done || self.last_try.elapsed() < RETRY {
                return None;
            }
            self.last_try = Instant::now();
            set_gpu_class(process, GPU_BELOW_NORMAL).ok()?;
            self.gpu_done = true;
            Some(format!("DECANT_GENTLE: GPU scheduling priority below normal, {:.1} s after the start", self.started.elapsed().as_secs_f64()))
        }
    }
}

#[cfg(windows)]
pub use platform::{Gentle, gpu_class, set_gpu_class};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_off_unless_set_to_something_true() {
        assert!(!enabled(None));
        for off in ["", " ", "0", "off", "False", "NO"] {
            assert!(!enabled(Some(off)), "{off:?}");
        }
        for on in ["1", "on", "true", "yes", "2"] {
            assert!(enabled(Some(on)), "{on:?}");
        }
    }

    #[test]
    fn the_line_names_what_was_applied_and_what_was_not() {
        let line = describe(&Ok(()), &Err("status 0xc000000d".into()));
        assert_eq!(line, "DECANT_GENTLE: CPU priority below normal, GPU scheduling priority not applied (status 0xc000000d), memory priority left alone");
    }
}
