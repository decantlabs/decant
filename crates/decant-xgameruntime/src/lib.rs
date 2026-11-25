//! `xgameruntime.dll` interposer: logs what a title asks the game runtime, then hands every call to
//! the PC runtime (Gaming Services' `xgameruntime.dll`), which already exports the same entry
//! points (`InitializeApiImpl`, `InitializeApiImplEx`, `InitializeApiImplEx2`, `QueryApiImpl`,
//! `UninitializeApiImpl`, `XErrorReport`, `DllCanUnloadNow`, checked on the public GDK 2610 and on
//! System32). Placed next to the executable, it would be loaded before the system copy. A port of
//! the former `src/gdk/xgameruntime/decant_xgr.c`, with the same exports, ordinals and log lines.
//!
//! Nothing is faked: a call the PC runtime refuses keeps the PC result. When the PC runtime is
//! missing (Windows Sandbox has no Gaming Services), `InitializeApiImpl*` accept the
//! initialisation and log it as such, and the other entries return `E_NOTIMPL`. These tables are
//! served locally, with or without a PC runtime (`DECANT_XGR_REAL_USER=1` forwards them when there
//! is one), so that no table the title uses gets our user handle or our queues into the PC
//! runtime:
//! * XAsync / XTaskQueue / XThread (`{073B7DCB-...}`, `local_table`, `taskqueue`, `xasync`);
//! * XUser and its gamertag interface (`{01ACD177-...}`, `local_user`): one local offline user,
//!   signed in, with no store account, no Xbox Live token, no gamer picture and every privilege;
//! * XNetworking (`{37E56907-...}`, `local_networking`): offline;
//! * XDisplay HDR (`{03F0FE74-...}`, `local_display`): HDR unavailable;
//! * XPackage, XGame, XSystem and its analytics, Accessibility (`local_package`, `local_game`,
//!   `local_system`, `local_accessibility`, with the title from `local_title`);
//! * XGameSave (`{704C3F58-...}`, `local_gamesave`, `local_gamesave_store`): saves in a per user
//!   folder, each update a new version committed with one rename.
//! * XPersistentLocalStorage (`{F4FAF4D4-...}`, `local_pls`): a host folder with the title's
//!   quota, under the drive `gdk-launch` maps for the run.
//! * XStore (`{0DD112AC-...}`, `local_store`): the offline answer for an owned retail title, a
//!   valid game licence, no add-on, empty product queries, nothing bought and no store contacted.
//!
//! Every other table is forwarded to the PC runtime, or refused with `REGDB_E_CLASSNOTREG` and a
//! null object when there is none. Every call on a local table is logged,
//! bounded, with a heartbeat every 5 s (`calllog`), so a title that hangs shows where it waits.
//! The log tells which API tables (`api_id` GUIDs) the title asks for and which the PC runtime
//! refuses. Known console tables are named from [`tables`] by the `(api_id, riid)` couple.
//!
//! Environment:
//! * `DECANT_XGR_REAL`: full path of the runtime to forward to (default `System32\xgameruntime.dll`).
//! * `DECANT_XGR_LOG`: file to append the log to (default: `OutputDebugString` only).
//! * `DECANT_XGR_TRACE=1`: every call on a PC runtime table is logged (`runtime_proxy`).
//! * `DECANT_XGR_PROXY=0`: PC runtime tables are handed out without the logging proxy.
//!
//! The log wording is plain functions here. The exports and the Windows calls behind them are in
//! `exports`; the local tables are in the `local_*` modules, `taskqueue`, `xasync` and
//! `calllog`. `console_mode` answers the title's registry read of the
//! console model as a Series X does. Only these modules have `unsafe` code.

#[cfg(windows)]
mod calllog;
#[cfg(windows)]
mod console_mode;
#[cfg(windows)]
mod exports;
#[cfg(windows)]
mod local_store;
#[cfg(windows)]
mod local_system;
#[cfg(windows)]
mod local_table;
#[cfg(windows)]
mod local_accessibility;
#[cfg(windows)]
mod local_display;
mod local_networking;
#[cfg(windows)]
mod local_game;
#[cfg(windows)]
mod local_gamesave;
#[cfg(windows)]
mod local_gamesave_store;
#[cfg(windows)]
mod local_package;
#[cfg(windows)]
mod local_pls;
#[cfg(windows)]
mod local_title;
#[cfg(windows)]
mod local_user;
#[cfg(all(windows, target_arch = "x86_64"))]
mod runtime_proxy;
#[cfg(all(windows, target_arch = "x86_64"))]
mod runtime_proxy_async;
pub mod tables;
#[cfg(windows)]
mod taskqueue;
#[cfg(windows)]
mod xasync;

use tables::{Guid, KnownTable, Level};

/// The entry points the interposer forwards, in the order of the C `enum`.
pub const NAMES: [&str; 7] =
    ["InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2", "QueryApiImpl", "UninitializeApiImpl", "XErrorReport", "DllCanUnloadNow"];

/// `E_NOTIMPL`.
pub const E_NOTIMPL: i32 = 0x8000_4001_u32 as i32;
/// `REGDB_E_CLASSNOTREG`: `QueryApiImpl`'s answer for a table it cannot serve.
pub const REGDB_E_CLASSNOTREG: i32 = 0x8004_0154_u32 as i32;
/// Appended to the line of an initialisation accepted without a PC runtime.
pub const NO_RUNTIME_INIT: &str = "(no PC runtime: initialisation accepted, only the local tables are served: XAsync/XTaskQueue/XThread, XUser, XNetworking, XDisplay HDR, XPackage, XGame, XSystem, Accessibility, XGameSave, XPersistentLocalStorage and XStore, every other table is refused)";
/// `S_OK`.
pub const S_OK: i32 = 0;
/// `E_POINTER`: `QueryApiImpl` without a PC runtime, for the local table and an unwritable `out`.
pub const E_POINTER: i32 = 0x8000_4003_u32 as i32;
/// Appended to the `QueryApiImpl` line when a local table was served, with the table's name.
pub fn local_table_note(table: &str) -> String {
    format!(", local {table} table (no PC runtime)")
}
/// `S_FALSE`.
pub const S_FALSE: i32 = 1;

/// Longest log line the C buffer held, with its prefix and without the final newline.
const LINE_MAX: usize = 1023;

/// The line written for one message: `[decant_xgr] `, the message cut to the C buffer size, and a
/// newline.
pub fn log_line(message: &str) -> String {
    let mut line = format!("[decant_xgr] {message}");
    let mut end = line.len().min(LINE_MAX);
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    line.truncate(end);
    line.push('\n');
    line
}

/// A GUID as the C code logged it: `{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}`, upper case.
pub fn guid_text(g: &Guid) -> String {
    let d = g.data4;
    format!("{{{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}}}", g.data1, g.data2, g.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7])
}

/// The text of a GUID argument, `-` when the argument is not a readable GUID.
fn guid_arg(g: Option<&Guid>) -> String {
    g.map_or_else(|| "-".to_string(), guid_text)
}

/// The line for a pass through call: the first four arguments, the first two read as GUIDs, and
/// the result (`None` when there is no PC runtime to call).
pub fn pass_line(name: &str, args: [usize; 4], arg1: Option<&Guid>, arg2: Option<&Guid>, result: Option<usize>) -> String {
    let head = format!("{name}(0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}) arg1-as-guid {} arg2-as-guid {}", args[0], args[1], args[2], args[3], guid_arg(arg1), guid_arg(arg2));
    match result {
        None => format!("{head} -> E_NOTIMPL (no PC runtime)"),
        Some(r) => format!("{head} -> 0x{:08x}", r as u32),
    }
}

/// What the `QueryApiImpl` call looked like: the GUIDs when readable, and the PC runtime's
/// answer.
pub struct Query<'a> {
    pub api: Option<&'a Guid>,
    pub riid: Option<&'a Guid>,
    pub result: i32,
    /// The table pointer the call left in `*out`, 0 when `out` was not readable.
    pub table: usize,
    /// False when there was no PC runtime to forward to.
    pub forwarded: bool,
    /// The name of the local table served (no PC runtime), `None` when none was.
    pub local: Option<&'a str>,
}

/// The lines for a `QueryApiImpl` call: the call itself, then, when a console table the PC
/// runtime refused, a second line.
pub fn query_lines(q: &Query) -> Vec<String> {
    let known: Option<(&KnownTable, Level)> = q.api.and_then(|api| tables::known(api, q.riid));
    let mut version = String::new();
    if let Some((table, level)) = known {
        version = format!(" = {}", if level == Level::Family { "version not listed for this family" } else { table.iid_note });
        if level == Level::Partial {
            version.push_str(", matched on the first field only");
        }
    }
    let name = known.map_or("", |(table, _)| table.name);
    let mut lines = vec![format!(
        "QueryApiImpl(api {}{}{}{}, riid {}{}) -> 0x{:08x}, table {:016X}{}",
        guid_arg(q.api),
        if known.is_some() { " = " } else { "" },
        name,
        if known.is_some() { " [hypothesis]" } else { "" },
        guid_arg(q.riid),
        version,
        q.result as u32,
        q.table,
        match q.local {
            Some(table) => local_table_note(table),
            None if q.forwarded => String::new(),
            None => " (no PC runtime)".to_string(),
        }
    )];
    if q.result < 0
        && let Some((table, _)) = known
    {
        lines.push(format!("  the PC runtime refused a table a console title uses ({}); no local table for this couple, see src/gdk/README.md", table.name));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASYNC: Guid = Guid { data1: 0x073B7DCB, data2: 0x1FCF, data3: 0x4030, data4: [0x94, 0xBE, 0xE3, 0xC9, 0xEB, 0x62, 0x34, 0x28] };
    const OTHER: Guid = Guid { data1: 0, data2: 0, data3: 0, data4: [0xC0, 0, 0, 0, 0, 0, 0, 0x46] };

    #[test]
    fn a_log_line_has_the_prefix_and_one_newline() {
        assert_eq!(log_line("x"), "[decant_xgr] x\n");
        assert_eq!(log_line(&"a".repeat(2000)).len(), LINE_MAX + 1);
    }

    #[test]
    fn a_guid_is_upper_case_in_braces() {
        assert_eq!(guid_text(&ASYNC), "{073B7DCB-1FCF-4030-94BE-E3C9EB623428}");
        assert_eq!(guid_text(&OTHER), "{00000000-0000-0000-C000-000000000046}");
    }

    #[test]
    fn a_pass_line_shows_the_arguments_and_the_result() {
        assert_eq!(
            pass_line("InitializeApiImplEx2", [1, 2, 3, 4], None, None, None),
            "InitializeApiImplEx2(0x1, 0x2, 0x3, 0x4) arg1-as-guid - arg2-as-guid - -> E_NOTIMPL (no PC runtime)"
        );
        assert_eq!(
            pass_line("XErrorReport", [0xdead, 0, 0, 0], Some(&ASYNC), None, Some(0x8000_4001)),
            "XErrorReport(0xdead, 0x0, 0x0, 0x0) arg1-as-guid {073B7DCB-1FCF-4030-94BE-E3C9EB623428} arg2-as-guid - -> 0x80004001"
        );
    }

    #[test]
    fn a_query_without_runtime_names_the_family_and_the_refusal() {
        let lines = query_lines(&Query { api: Some(&ASYNC), riid: Some(&OTHER), result: E_NOTIMPL, table: 1, forwarded: false, local: None });
        assert_eq!(
            lines[0],
            "QueryApiImpl(api {073B7DCB-1FCF-4030-94BE-E3C9EB623428} = XAsync / XTaskQueue / XThread (32 slots) [hypothesis], \
             riid {00000000-0000-0000-C000-000000000046} = version not listed for this family) -> 0x80004001, table 0000000000000001 (no PC runtime)"
        );
        assert_eq!(
            lines[1],
            "  the PC runtime refused a table a console title uses (XAsync / XTaskQueue / XThread (32 slots)); no local table for this couple, see src/gdk/README.md"
        );
    }

    #[test]
    fn a_listed_riid_and_an_unknown_api_are_logged() {
        let lines = query_lines(&Query { api: Some(&ASYNC), riid: Some(&ASYNC), result: 0, table: 0xABC, forwarded: true, local: None });
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("riid {073B7DCB-1FCF-4030-94BE-E3C9EB623428} = riid = api_id) -> 0x00000000, table 0000000000000ABC"));
        let lines = query_lines(&Query { api: Some(&OTHER), riid: None, result: E_NOTIMPL, table: 0, forwarded: true, local: None });
        assert_eq!(lines, ["QueryApiImpl(api {00000000-0000-0000-C000-000000000046}, riid -) -> 0x80004001, table 0000000000000000"]);
    }

    #[test]
    fn the_local_table_is_logged_plainly() {
        let lines = query_lines(&Query { api: Some(&ASYNC), riid: Some(&ASYNC), result: S_OK, table: 0x7FF0_1234, forwarded: false, local: Some("XAsync/XTaskQueue/XThread") });
        assert_eq!(
            lines,
            ["QueryApiImpl(api {073B7DCB-1FCF-4030-94BE-E3C9EB623428} = XAsync / XTaskQueue / XThread (32 slots) [hypothesis], riid {073B7DCB-1FCF-4030-94BE-E3C9EB623428} = riid = api_id) -> 0x00000000, table 000000007FF01234, local XAsync/XTaskQueue/XThread table (no PC runtime)"]
        );
    }
}
