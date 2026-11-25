//! The log of the calls made on the local tables (served only without a PC runtime), so that a
//! title that hangs shows in the log where it waits: every slot call logs one line with its
//! arguments before it runs and one line with its result after it returned. To keep the log
//! bounded, a slot logs its first [`FIRST`] calls, then every [`EVERY`]th. A heartbeat thread,
//! started when the first local table is served, logs every [`HEARTBEAT_SECS`] s the live task
//! queues, the callbacks they still hold and the async operations still active.
//!
//! Our own code.

#![allow(unsafe_code)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleHandleExW};

use crate::exports::log;
use crate::taskqueue::RegistrationToken;

/// Every call of a slot up to this one is logged.
pub(crate) const FIRST: u64 = 200;
/// After [`FIRST`], every call whose number is a multiple of this one is logged.
pub(crate) const EVERY: u64 = 1000;
/// The heartbeat period.
pub(crate) const HEARTBEAT_SECS: u64 = 5;

/// Whether call number `n` (from 1) of a slot is logged.
pub(crate) fn logged(n: u64) -> bool {
    n <= FIRST || n.is_multiple_of(EVERY)
}

/// One call counter per slot of a table.
pub(crate) struct Counters<const N: usize>([AtomicU64; N]);

impl<const N: usize> Counters<N> {
    pub(crate) const fn new() -> Self {
        Counters([const { AtomicU64::new(0) }; N])
    }

    /// Counts a call of `slot`; its number when the call is to be logged.
    pub(crate) fn next(&self, slot: usize) -> Option<u64> {
        let n = self.0.get(slot)?.fetch_add(1, Ordering::Relaxed) + 1;
        logged(n).then_some(n)
    }
}

/// An argument or a result as the log shows it: pointers and callbacks in hex, modes, ports and
/// sizes in decimal, `HRESULT`s on eight hex digits.
pub(crate) trait Arg {
    fn text(&self) -> String;
}

impl<T> Arg for *mut T {
    fn text(&self) -> String {
        format!("0x{:x}", *self as usize)
    }
}

impl<T> Arg for *const T {
    fn text(&self) -> String {
        format!("0x{:x}", *self as usize)
    }
}

impl Arg for usize {
    fn text(&self) -> String {
        format!("0x{self:x}")
    }
}

impl Arg for u64 {
    fn text(&self) -> String {
        format!("0x{self:x}")
    }
}

impl Arg for u32 {
    fn text(&self) -> String {
        self.to_string()
    }
}

impl Arg for i32 {
    fn text(&self) -> String {
        format!("0x{:08x}", *self as u32)
    }
}

impl Arg for bool {
    fn text(&self) -> String {
        self.to_string()
    }
}

impl Arg for RegistrationToken {
    fn text(&self) -> String {
        format!("0x{:x}", self.token)
    }
}

impl Arg for () {
    fn text(&self) -> String {
        "done".to_string()
    }
}

/// The function name of a path as `stringify!` writes it (`xa :: XAsyncBegin` -> `XAsyncBegin`).
fn short(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path).trim()
}

/// The line logged before a slot call.
pub(crate) fn call_line(table: &str, slot: usize, function: &str, args: &[(&str, String)], n: u64) -> String {
    let args: Vec<String> = args.iter().map(|(name, value)| format!("{name} {value}")).collect();
    format!("local {table} table: slot {slot} {}({}) [call {n}]", short(function), args.join(", "))
}

/// The line logged after a slot call returned.
pub(crate) fn return_line(table: &str, slot: usize, function: &str, result: &str, n: u64) -> String {
    format!("local {table} table: slot {slot} {} -> {result} [call {n}]", short(function))
}

/// Declares a slot that logs the call (bounded by `$counts`), drops `this`, calls the table
/// function with the other arguments and logs its result. The log names the slot by its GDK
/// function, `as "Name"` when the function behind it has another name.
macro_rules! logged_shim {
    ($table:expr, $counts:expr, $this:ty, $slot:literal as $gdk:literal, $name:ident => $target:path, $args:tt $(-> $ret:ty)?) => {
        $crate::calllog::logged_shim!(@slot $table, $counts, $this, $slot, $gdk, $name => $target, $args $(-> $ret)?);
    };
    ($table:expr, $counts:expr, $this:ty, $slot:literal, $name:ident => $target:path, $args:tt $(-> $ret:ty)?) => {
        $crate::calllog::logged_shim!(@slot $table, $counts, $this, $slot, stringify!($target), $name => $target, $args $(-> $ret)?);
    };
    (@slot $table:expr, $counts:expr, $this:ty, $slot:literal, $gdk:expr, $name:ident => $target:path, ($($arg:ident: $ty:ty),*) $(-> $ret:ty)?) => {
        #[allow(clippy::let_unit_value)]
        unsafe extern "system" fn $name(_this: $this, $($arg: $ty),*) $(-> $ret)? {
            let n = $counts.next($slot);
            if let Some(n) = n {
                $crate::exports::log(&$crate::calllog::call_line($table, $slot, $gdk, &[$((stringify!($arg), $crate::calllog::Arg::text(&$arg))),*], n));
            }
            // SAFETY: the title passes the arguments of the GDK function, unchanged.
            let r = unsafe { $target($($arg),*) };
            if let Some(n) = n {
                $crate::exports::log(&$crate::calllog::return_line($table, $slot, $gdk, &$crate::calllog::Arg::text(&r), n));
            }
            r
        }
    };
}
pub(crate) use logged_shim;

/// Starts the heartbeat thread once.
pub(crate) fn start_heartbeat() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    // The thread runs this module's code until the process ends: pin the module, so that a
    // `FreeLibrary` of the interposer never unmaps it under the thread.
    let mut module = std::ptr::null_mut();
    // SAFETY: the address is a function of this module; `module` is a writable handle location.
    unsafe {
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN, start_heartbeat as *const () as *const u16, &mut module);
    }
    let started = std::time::Instant::now();
    let spawned = std::thread::Builder::new().name("decant-xgr-heartbeat".into()).spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(HEARTBEAT_SECS));
            let (queues, callbacks) = crate::taskqueue::stats();
            log(&heartbeat_line(started.elapsed().as_secs(), queues, callbacks, crate::xasync::active_count()));
        }
    });
    if spawned.is_err() {
        log("heartbeat: the thread could not be started");
    }
}

/// The heartbeat line.
pub(crate) fn heartbeat_line(secs: u64, queues: usize, callbacks: usize, async_ops: usize) -> String {
    format!("heartbeat t={secs}s: {queues} queues, {callbacks} pending callbacks, {async_ops} pending async ops")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_logs_its_first_200_calls_then_every_1000th() {
        let logged_calls: Vec<u64> = (1..=5000).filter(|&n| logged(n)).collect();
        assert_eq!(logged_calls.len(), 200 + 5);
        assert_eq!(&logged_calls[198..], [199, 200, 1000, 2000, 3000, 4000, 5000]);
        assert!(!logged(201) && !logged(999) && logged(1000) && !logged(1001));

        let counts = Counters::<3>::new();
        let seen: Vec<Option<u64>> = (0..1000).map(|_| counts.next(1)).collect();
        assert_eq!(seen[0], Some(1));
        assert_eq!(seen[199], Some(200));
        assert!(seen[200..999].iter().all(Option::is_none));
        assert_eq!(seen[999], Some(1000));
        assert_eq!(counts.next(0), Some(1), "each slot counts on its own");
        assert_eq!(counts.next(3), None, "a slot out of the table is not counted");
    }

    #[test]
    fn the_lines_name_the_slot_the_arguments_and_the_result() {
        let args = [("queue", std::ptr::null_mut::<std::ffi::c_void>().text()), ("port", 1u32.text()), ("cb", 0xABCusize.text()), ("wait", true.text())];
        assert_eq!(
            call_line("XAsync/XTaskQueue/XThread", 16, "tq :: XTaskQueueDispatch", &args, 3),
            "local XAsync/XTaskQueue/XThread table: slot 16 XTaskQueueDispatch(queue 0x0, port 1, cb 0xabc, wait true) [call 3]"
        );
        assert_eq!(return_line("XUser", 7, "XUserAddAsync", &0i32.text(), 1), "local XUser table: slot 7 XUserAddAsync -> 0x00000000 [call 1]");
        assert_eq!(().text(), "done");
        assert_eq!(heartbeat_line(10, 2, 3, 1), "heartbeat t=10s: 2 queues, 3 pending callbacks, 1 pending async ops");
    }
}
