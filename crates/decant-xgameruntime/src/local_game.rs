//! Offline XGame table: `api_id = {973A344E-24BF-4D0F-8457-56C534892B29}`, interface
//! `{2549F142-6419-4A06-97B5-931AAB7C2F34}`. For an unregistered exe the PC runtime knows no title
//! id, and its `XLaunchNewGame` does not end the process (`decant-specs/hle-gap.md` row 8).
//!
//! * Slot 3 `XGameGetXboxTitleId` returns the title id of `MicrosoftGame.config` (`local_title`).
//! * Slot 4 `XLaunchNewGame(exe, args, user)`. Some titles restart themselves through it, passing
//!   their own path and arguments from their configuration, and on a console the call never
//!   returns. The call is logged with its arguments, then the process ends. When `decant-launch`
//!   started the title it sets [`RELAUNCH_FILE_ENV`]. The request is then written to that file and
//!   the process exits with [`RELAUNCH_EXIT_CODE`], so the launcher starts the title again with
//!   the new arguments, under its debugger and with its game drive. Without a launcher the new
//!   process is started right here before this one ends.
//! * Slot 5 `XLaunchRestartOnCrash` is accepted and logged, no restart is registered.
//!
//! Slots 6 to 11 log a named "not implemented" (`local_title`). Every call is logged with a bound
//! (`calllog`). Signatures come from the public GDK `XGame.h` and `XLauncher.h`.
#![allow(unsafe_code)]

use std::ffi::{OsString, c_char, c_void};
use std::path::PathBuf;

use windows_sys::Win32::System::Threading::ExitProcess;

use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::local_title::{LocalTable, MAX_SLOTS, Methods, Object, no_count, not_implemented, put, query_interface, slot, text, title};
use crate::tables::Guid;
use crate::taskqueue::S_OK;

/// The table's `api_id`.
pub(crate) const XGAME_TABLE: Guid = Guid { data1: 0x973A344E, data2: 0x24BF, data3: 0x4D0F, data4: [0x84, 0x57, 0x56, 0xC5, 0x34, 0x89, 0x2B, 0x29] };
/// The interface the title asks for.
pub(crate) const XGAME_IID: Guid = Guid { data1: 0x2549F142, data2: 0x6419, data3: 0x4A06, data4: [0x97, 0xB5, 0x93, 0x1A, 0xAB, 0x7C, 0x2F, 0x34] };

/// The variable `decant-launch` sets to the file a relaunch request goes to (`decant-launch`'s
/// `relaunch::FILE_ENV`, same value).
pub(crate) const RELAUNCH_FILE_ENV: &str = "DECANT_RELAUNCH_FILE";
/// The exit code that asks `decant-launch` for a relaunch ("SXRL", `decant-launch`'s
/// `relaunch::EXIT_CODE`, same value).
pub(crate) const RELAUNCH_EXIT_CODE: u32 = 0x5358_524C;

const NAME: &str = "XGame";
const SLOTS: usize = 12;
static COUNTS: Counters<MAX_SLOTS> = Counters::new();
static OBJECT: Object = Object::new(&VTBL);

/// The object for the pair this table serves, with its name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    (api == Some(&XGAME_TABLE) && riid == Some(&XGAME_IID)).then(|| (OBJECT.as_out(), NAME))
}

struct Game;

impl LocalTable for Game {
    const NAME: &'static str = NAME;
    const IIDS: &'static [Guid] = &[XGAME_IID];
    fn counts() -> &'static Counters<MAX_SLOTS> {
        &COUNTS
    }
    fn slot_name(slot: usize) -> Option<&'static str> {
        match slot {
            3 => Some("XGameGetXboxTitleId"),
            4 => Some("XLaunchNewGame"),
            5 => Some("XLaunchRestartOnCrash"),
            _ => None,
        }
    }
    fn object() -> *mut c_void {
        OBJECT.as_out()
    }
}

type This = *const c_void;

logged_shim!(NAME, COUNTS, This, 3 as "XGameGetXboxTitleId", title_id_entry => report_title_id, (out: *mut u32) -> i32);
logged_shim!(NAME, COUNTS, This, 4 as "XLaunchNewGame", launch_new_game_entry => relaunch_and_exit, (exe: *const c_char, args: *const c_char, user: *mut c_void));
logged_shim!(NAME, COUNTS, This, 5 as "XLaunchRestartOnCrash", restart_on_crash_entry => note_restart_request, (args: *const c_char, reserved: u32) -> i32);

unsafe fn report_title_id(out: *mut u32) -> i32 {
    put(out, title().id)
}

/// What a relaunch does.
#[derive(Debug, PartialEq, Eq)]
enum Relaunch {
    /// Write the request to the launcher's file, then end with [`RELAUNCH_EXIT_CODE`].
    Launcher { file: PathBuf, request: String },
    /// No launcher: start the executable with the arguments here, then end.
    Direct { exe: PathBuf, args: String },
}

/// The request `decant-launch` reads: the executable on the first line, the arguments as one
/// command line tail on the second.
fn launcher_request(exe: &str, args: &str) -> String {
    format!("{}\n{}\n", exe.trim(), args.trim())
}

/// The relaunch for `exe` and `args`, with the launcher's file variable `file` when it is set.
/// An empty `exe` means this executable.
fn plan_relaunch(exe: &str, args: &str, file: Option<OsString>, current: PathBuf) -> Relaunch {
    match file.filter(|f| !f.is_empty()) {
        Some(file) => Relaunch::Launcher { file: PathBuf::from(file), request: launcher_request(exe, args) },
        None => Relaunch::Direct { exe: if exe.trim().is_empty() { current } else { PathBuf::from(exe.trim()) }, args: args.trim().to_owned() },
    }
}

/// Carries out a relaunch plan and returns the exit code for this process.
fn hand_over(plan: Relaunch) -> u32 {
    match plan {
        Relaunch::Launcher { file, request } => {
            match std::fs::write(&file, request) {
                Ok(()) => log(&format!("XLaunchNewGame: request written to {}, exiting with {RELAUNCH_EXIT_CODE:#010x} so the launcher relaunches", file.display())),
                Err(e) => log(&format!("XLaunchNewGame: the request could not be written to {} ({e}), exiting without a relaunch", file.display())),
            }
            RELAUNCH_EXIT_CODE
        }
        Relaunch::Direct { exe, args } => {
            use std::os::windows::process::CommandExt;
            let mut command = std::process::Command::new(&exe);
            command.raw_arg(&args);
            if let Some(dir) = exe.parent().filter(|d| !d.as_os_str().is_empty()) {
                command.current_dir(dir);
            }
            match command.spawn() {
                Ok(child) => log(&format!("XLaunchNewGame: no launcher ({RELAUNCH_FILE_ENV} not set), started {} as process {}, exiting", exe.display(), child.id())),
                Err(e) => log(&format!("XLaunchNewGame: no launcher and {} did not start ({e}), exiting", exe.display())),
            }
            0
        }
    }
}

/// Logs the request, hands the relaunch over and ends the process. It never returns, as on a console.
unsafe fn relaunch_and_exit(exe: *const c_char, args: *const c_char, _user: *mut c_void) {
    let (exe, args) = (text(exe).unwrap_or_default(), text(args).unwrap_or_default());
    log(&format!("XLaunchNewGame: the title asks to start again: exe {exe:?}, args {args:?}"));
    let current = std::env::current_exe().unwrap_or_default();
    let code = hand_over(plan_relaunch(&exe, &args, std::env::var_os(RELAUNCH_FILE_ENV), current));
    // SAFETY: ends the process, as XLaunchNewGame does on a console.
    unsafe { ExitProcess(code) }
}

unsafe fn note_restart_request(args: *const c_char, _reserved: u32) -> i32 {
    log(&format!("XLaunchRestartOnCrash: args {:?} noted, no restart is registered", text(args).unwrap_or_default()));
    S_OK
}

static VTBL: Methods<SLOTS> = Methods([
    slot!(query_interface::<Game>),
    slot!(no_count),
    slot!(no_count),
    slot!(title_id_entry),
    slot!(launch_new_game_entry),
    slot!(restart_on_crash_entry),
    slot!(not_implemented::<Game, 6>),
    slot!(not_implemented::<Game, 7>),
    slot!(not_implemented::<Game, 8>),
    slot!(not_implemented::<Game, 9>),
    slot!(not_implemented::<Game, 10>),
    slot!(not_implemented::<Game, 11>),
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::E_NOTIMPL;
    use crate::exports::query_without_runtime;
    use crate::local_title::method;

    fn table() -> *mut c_void {
        let mut out = std::ptr::null_mut();
        assert_eq!(query_without_runtime(Some(&XGAME_TABLE), Some(&XGAME_IID), &mut out), (S_OK, Some(NAME)));
        out
    }

    #[test]
    fn the_title_id_comes_from_the_config() {
        let t = table();
        type Get = unsafe extern "system" fn(This, *mut u32) -> i32;
        type Restart = unsafe extern "system" fn(This, *const c_char, u32) -> i32;
        type Unknown = unsafe extern "system" fn(This) -> i32;
        let mut id = 0xDEAD;
        // SAFETY: the slots get their GDK arguments.
        unsafe {
            assert_eq!(method::<Get>(t, 3)(t, &mut id), S_OK);
            assert_eq!(id, title().id);
            assert_eq!(method::<Restart>(t, 5)(t, c"-restart".as_ptr(), 0), S_OK);
            for slot in 6..SLOTS {
                assert_eq!(method::<Unknown>(t, slot)(t), E_NOTIMPL);
            }
        }
        assert!(object_for(Some(&XGAME_TABLE), Some(&XGAME_TABLE)).is_none());
    }

    #[test]
    fn a_relaunch_goes_to_the_launcher_when_it_listens() {
        let current = PathBuf::from(r"D:\title_overlay\title.exe");
        assert_eq!(
            plan_relaunch(r"D:\title_overlay\title.exe", " -a -b=\"x y\" ", Some(OsString::from(r"D:\o\decant_relaunch.txt")), current.clone()),
            Relaunch::Launcher { file: PathBuf::from(r"D:\o\decant_relaunch.txt"), request: "D:\\title_overlay\\title.exe\n-a -b=\"x y\"\n".to_owned() }
        );
        assert_eq!(plan_relaunch("", "-a", None, current.clone()), Relaunch::Direct { exe: current.clone(), args: "-a".to_owned() });
        assert_eq!(plan_relaunch("other.exe", "", Some(OsString::new()), current), Relaunch::Direct { exe: PathBuf::from("other.exe"), args: String::new() });
        assert_eq!(RELAUNCH_EXIT_CODE.to_be_bytes(), *b"SXRL");
    }
}
