//! For `DECANT_WAITS`: a Decant DLL that creates an event or a semaphore a title thread may wait on
//! says so with [`created`]. The line goes to the debugger (`[decant-handle] <handle> <dll>: <what>,
//! called from <module+rva>`), and `decant-launch` names the creator when it finds a thread that waits
//! on that handle for long. Off (the default), [`created`] does nothing but one atomic load.

use std::sync::atomic::{AtomicU8, Ordering};

/// The switch of the launcher's wait report, read here too.
pub const ENV: &str = "DECANT_WAITS";
/// The prefix of the debug line.
pub const PREFIX: &str = "[decant-handle] ";

static STATE: AtomicU8 = AtomicU8::new(0);

fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var(ENV).is_ok_and(|v| !v.trim().is_empty() && v.trim() != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// The debug line for `handle`, made by `module` for `what`, asked for by `caller`.
pub fn line(handle: usize, module: &str, what: &str, caller: &str) -> String {
    format!("{PREFIX}{handle:#x} {module}: {what}, called from {caller}")
}

/// Parses a line made by [`line`]: the handle and the rest.
pub fn parse(text: &str) -> Option<(usize, String)> {
    let rest = text.trim().strip_prefix(PREFIX)?;
    let (handle, who) = rest.split_once(' ')?;
    let handle = usize::from_str_radix(handle.strip_prefix("0x")?, 16).ok()?;
    Some((handle, who.to_owned()))
}

/// Reports that this DLL created `handle` (an event, a semaphore) for `what`.
#[inline]
pub fn created(handle: usize, what: &str) {
    if enabled() {
        report(handle, what);
    }
}

#[cold]
#[inline(never)]
fn report(handle: usize, what: &str) {
    #[cfg(windows)]
    {
        let module = crate::calls::module_of(report as *const () as usize).map_or_else(|| "?".to_owned(), |(_, n)| n);
        crate::debug_string(&line(handle, &module, what, &crate::calls::caller_outside_this_module()));
    }
    #[cfg(not(windows))]
    let _ = (handle, what);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_reads_back() {
        let l = line(0x1a4, "d3d12_xs.dll", "fence event", "title.exe+0x10");
        assert_eq!(l, "[decant-handle] 0x1a4 d3d12_xs.dll: fence event, called from title.exe+0x10");
        assert_eq!(parse(&l), Some((0x1a4, "d3d12_xs.dll: fence event, called from title.exe+0x10".into())));
        assert_eq!(parse("d3d12_xs: something else"), None);
    }
}
