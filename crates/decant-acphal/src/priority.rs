//! Priority of the audio threads of this DLL (the XMA service thread, the monitor that plays what it decodes).
//!
//! They ran at the default priority with the default 15.6 ms timer. While a title loads its world, its many
//! threads keep every core busy: the service thread that fills the XMA output rings then came late, and the
//! title's time-critical mixer thread, which sends its commands under the same lock, waited for it (audio
//! crackle, then cuts). A raised priority and the "Pro Audio" task of the multimedia scheduler keep both
//! threads running within their 2 ms and 10 ms periods. `DECANT_AUDIO_PRIORITY=0` leaves them as they were.
#![allow(unsafe_code)]

/// Whether the audio threads get a raised priority (`DECANT_AUDIO_PRIORITY=0` turns it off).
pub fn wanted() -> bool {
    enabled(std::env::var("DECANT_AUDIO_PRIORITY").ok().as_deref())
}

/// Whether this value of the variable keeps the raised priority: anything but `0` (pure, for the tests).
pub fn enabled(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// Raises the calling thread for audio work. Returns what was done, for the log.
#[cfg(windows)]
pub fn raise_current_thread() -> String {
    use windows::Win32::Media::timeBeginPeriod;
    use windows::Win32::System::Threading::{AvSetMmThreadCharacteristicsW, GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST};
    use windows::core::w;
    if !wanted() {
        return "left at the default priority (DECANT_AUDIO_PRIORITY=0)".into();
    }
    // SAFETY: plain calls on the current thread; the 1 ms timer lasts for the process (the waits of 2 and 10 ms of these threads were 15.6 ms long at the default resolution).
    unsafe {
        static TIMER: std::sync::Once = std::sync::Once::new();
        TIMER.call_once(|| {
            timeBeginPeriod(1);
        });
        let priority = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST).is_ok();
        let mut task = 0u32;
        let mmcss = AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut task).is_ok_and(|h| !h.is_invalid());
        format!("timer 1 ms, priority highest {}, multimedia scheduler 'Pro Audio' {}", if priority { "set" } else { "refused" }, if mmcss { "joined" } else { "refused" })
    }
}

#[cfg(not(windows))]
pub fn raise_current_thread() -> String {
    "nothing to raise on this platform".into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_raised_priority_is_on_unless_the_variable_is_zero() {
        assert!(super::enabled(None) && super::enabled(Some("1")) && super::enabled(Some("")));
        assert!(!super::enabled(Some("0")) && !super::enabled(Some(" 0 ")));
    }
}
