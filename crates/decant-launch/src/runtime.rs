//! Whether Microsoft's game runtime is on this machine. On Windows the GDK API a title reaches
//! through `QueryApiImpl` (XUser, XStore, XPackage, XGameRuntime...) is served by Microsoft's
//! Gaming Services, whose runtime DLL is `xgameruntime.dll` in the system folder. The launcher
//! changes nothing there: it only reports what it finds, so a missing runtime explains a title that
//! stops right after `XGameRuntimeInitialize`. The installer sets Gaming Services up on the user's
//! machine.

use std::path::PathBuf;

/// The service of Microsoft's GameInput redistributable, which the title needs: without it
/// `GameInputCreate` fails after about five seconds.
pub const GAMEINPUT_SERVICE: &str = "GameInputSvc";

/// The services Gaming Services installs.
pub const SERVICES: [&str; 2] = ["GamingServices", "GamingServicesNet"];

/// The state of one Gaming Services service, as the service manager reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceState {
    Running,
    /// Installed and stopped: it starts on demand.
    Stopped,
    StartingOrStopping,
    /// Not installed.
    Missing,
    /// The service manager could not be asked, or did not answer.
    Unknown,
    /// Not on Windows.
    NotWindows,
}

impl ServiceState {
    /// The words of the `decant-launch` log.
    pub fn word(self) -> &'static str {
        match self {
            ServiceState::Running => "running",
            ServiceState::Stopped => "stopped (it starts on demand)",
            ServiceState::StartingOrStopping => "starting or stopping",
            ServiceState::Missing => "MISSING (install Gaming Services)",
            ServiceState::Unknown => "unknown",
            ServiceState::NotWindows => "not on Windows",
        }
    }
}

/// What was found: the runtime DLL and each service of [`SERVICES`], in that order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Runtime {
    pub dll: PathBuf,
    pub dll_present: bool,
    pub services: [ServiceState; 2],
}

impl Runtime {
    /// Present and every service installed (running or ready to start on demand).
    pub fn ready(&self) -> bool {
        self.dll_present && self.services.iter().all(|s| matches!(s, ServiceState::Running | ServiceState::Stopped | ServiceState::StartingOrStopping))
    }
}

/// Reads the runtime's state. Nothing is changed.
pub fn state() -> Runtime {
    let (runtime, _) = state_and_reachability();
    runtime
}

fn state_and_reachability() -> (Runtime, bool) {
    let system = std::env::var_os("SystemRoot").map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from).join("System32");
    let dll = system.join("xgameruntime.dll");
    let (services, reachable) = query_services();
    (Runtime { dll_present: dll.is_file(), dll, services }, reachable)
}

/// One line per fact: the runtime DLL, then each service and its state.
pub fn report() -> Vec<String> {
    let (runtime, reachable) = state_and_reachability();
    let mut out = vec![format!("runtime: {} {}", runtime.dll.display(), if runtime.dll_present { "present" } else { "MISSING (install Gaming Services)" })];
    for (name, state) in SERVICES.iter().zip(runtime.services) {
        let word = if state == ServiceState::Unknown && !reachable { "unknown (no access to the service manager)" } else { state.word() };
        out.push(format!("runtime: service {name} {word}"));
    }
    out
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod platform {
    use super::ServiceState;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::Services::{
        CloseServiceHandle, OpenSCManagerW, OpenServiceW, QueryServiceStatus, SC_MANAGER_CONNECT, SERVICE_QUERY_STATUS, SERVICE_RUNNING, SERVICE_STATUS,
        SERVICE_STOPPED,
    };

    /// The file version (major, minor, build, revision) from the version resource, if it has one.
    pub fn file_version(path: &std::path::Path) -> Option<[u16; 4]> {
        use windows_sys::Win32::Storage::FileSystem::{GetFileVersionInfoSizeW, GetFileVersionInfoW, VS_FIXEDFILEINFO, VerQueryValueW};
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let root: Vec<u16> = "\\".encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: the buffer is sized by the size query and outlives the pointer VerQueryValueW returns into it.
        unsafe {
            let mut handle = 0;
            let size = GetFileVersionInfoSizeW(wide.as_ptr(), &mut handle);
            if size == 0 {
                return None;
            }
            let mut data = vec![0u8; size as usize];
            if GetFileVersionInfoW(wide.as_ptr(), 0, size, data.as_mut_ptr().cast()) == 0 {
                return None;
            }
            let mut info: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut len = 0;
            if VerQueryValueW(data.as_ptr().cast(), root.as_ptr(), &mut info, &mut len) == 0 || info.is_null() || (len as usize) < std::mem::size_of::<VS_FIXEDFILEINFO>() {
                return None;
            }
            let fixed = std::ptr::read_unaligned(info.cast::<VS_FIXEDFILEINFO>());
            Some([(fixed.dwFileVersionMS >> 16) as u16, fixed.dwFileVersionMS as u16, (fixed.dwFileVersionLS >> 16) as u16, fixed.dwFileVersionLS as u16])
        }
    }

    fn query_with_manager(manager: windows_sys::Win32::System::Services::SC_HANDLE, name: &str) -> ServiceState {
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: `manager` is a valid service manager handle and `wide` is NUL-terminated.
        unsafe {
            let service = OpenServiceW(manager, wide.as_ptr(), SERVICE_QUERY_STATUS);
            if service.is_null() {
                ServiceState::Missing
            } else {
                let mut status: SERVICE_STATUS = std::mem::zeroed();
                let state = if QueryServiceStatus(service, &mut status) == 0 {
                    ServiceState::Unknown
                } else {
                    match status.dwCurrentState {
                        SERVICE_RUNNING => ServiceState::Running,
                        SERVICE_STOPPED => ServiceState::Stopped,
                        _ => ServiceState::StartingOrStopping,
                    }
                };
                CloseServiceHandle(service);
                state
            }
        }
    }

    pub fn query_services() -> ([ServiceState; 2], bool) {
        // SAFETY: plain service manager queries; every handle opened is closed.
        unsafe {
            let manager = OpenSCManagerW(std::ptr::null(), std::ptr::null(), SC_MANAGER_CONNECT);
            if manager.is_null() {
                return ([ServiceState::Unknown, ServiceState::Unknown], false);
            }
            let states = [
                query_with_manager(manager, super::SERVICES[0]),
                query_with_manager(manager, super::SERVICES[1]),
            ];
            CloseServiceHandle(manager);
            (states, true)
        }
    }

    pub fn service_state(name: &str) -> ServiceState {
        // SAFETY: plain service manager queries; every handle opened is closed.
        unsafe {
            let manager = OpenSCManagerW(std::ptr::null(), std::ptr::null(), SC_MANAGER_CONNECT);
            if manager.is_null() {
                return ServiceState::Unknown;
            }
            let state = query_with_manager(manager, name);
            CloseServiceHandle(manager);
            state
        }
    }
}

#[cfg(windows)]
use platform::query_services;
#[cfg(windows)]
pub use platform::{file_version, service_state};

/// The state of one service by name. Nothing is changed.
#[cfg(not(windows))]
pub fn service_state(_: &str) -> ServiceState {
    ServiceState::NotWindows
}

/// The file version (major, minor, build, revision), if the file has a version resource.
#[cfg(not(windows))]
pub fn file_version(_: &std::path::Path) -> Option<[u16; 4]> {
    None
}

#[cfg(not(windows))]
fn query_services() -> ([ServiceState; 2], bool) {
    ([ServiceState::NotWindows, ServiceState::NotWindows], true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_report_names_the_runtime_dll_and_both_services() {
        let lines = report();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("xgameruntime.dll"));
        assert!(lines[1].contains("GamingServices ") && lines[2].contains("GamingServicesNet"));
    }

    #[test]
    fn the_state_agrees_with_the_report() {
        let runtime = state();
        let lines = report();
        assert_eq!(lines[0].ends_with("present"), runtime.dll_present);
        for (line, s) in lines[1..].iter().zip(runtime.services) {
            assert!(line.contains(s.word().split(' ').next().unwrap()), "{line} for {s:?}");
        }
    }

    #[test]
    fn ready_needs_the_dll_and_every_service_installed() {
        let mut runtime = Runtime { dll: PathBuf::from("xgameruntime.dll"), dll_present: true, services: [ServiceState::Running, ServiceState::Stopped] };
        assert!(runtime.ready());
        runtime.services[1] = ServiceState::Missing;
        assert!(!runtime.ready());
        runtime.services[1] = ServiceState::Running;
        runtime.dll_present = false;
        assert!(!runtime.ready());
    }
}
