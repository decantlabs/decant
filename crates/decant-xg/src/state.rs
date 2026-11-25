//! The process wide state of the library (`docs/CONSOLE_LIBS.md`, section 1): the initialised flag, the hardware version flag and the
//! exclusive locking flag with its critical section.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

static INITIALIZED: AtomicBool = AtomicBool::new(false);
static HARDWARE_V4: AtomicBool = AtomicBool::new(false);
static LOCKING: AtomicBool = AtomicBool::new(false);

/// Sets the initialised flag. 0 when it was clear, 1 when it was already set.
pub fn initialize() -> i32 {
    i32::from(INITIALIZED.swap(true, Ordering::SeqCst))
}

/// Clears the initialised flag. 0 when it was set, 1 when it was already clear.
pub fn terminate() -> i32 {
    i32::from(!INITIALIZED.swap(false, Ordering::SeqCst))
}

/// `XGSetHardwareVersion`: version 4 selects the other address equations. Then it behaves like `initialize`.
pub fn set_hardware_version(version: u32) -> i32 {
    HARDWARE_V4.store(version == 4, Ordering::SeqCst);
    initialize()
}

/// Whether version 4 equations are in use.
pub fn hardware_v4() -> bool {
    HARDWARE_V4.load(Ordering::SeqCst)
}

/// A critical section: entered again by the thread that owns it.
struct CriticalSection {
    state: Mutex<(Option<ThreadId>, u32)>,
    released: Condvar,
}

static SECTION: CriticalSection = CriticalSection { state: Mutex::new((None, 0)), released: Condvar::new() };

impl CriticalSection {
    fn enter(&self) {
        let me = std::thread::current().id();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match state.0 {
                None => {
                    *state = (Some(me), 1);
                    return;
                }
                Some(owner) if owner == me => {
                    state.1 += 1;
                    return;
                }
                Some(_) => state = self.released.wait(state).unwrap_or_else(|e| e.into_inner()),
            }
        }
    }

    fn leave(&self) {
        let me = std::thread::current().id();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == Some(me) {
            state.1 -= 1;
            if state.1 == 0 {
                state.0 = None;
                self.released.notify_one();
            }
        }
    }
}

/// `XGEnableExclusiveLocking`.
pub fn enable_locking(enable: bool) {
    LOCKING.store(enable, Ordering::SeqCst);
}

/// `XGEnterExclusiveLock`: 1 after entering the critical section, 0 without waiting when locking is off.
pub fn enter_lock() -> i32 {
    if LOCKING.load(Ordering::SeqCst) {
        SECTION.enter();
        1
    } else {
        0
    }
}

/// `XGLeaveExclusiveLock`.
pub fn leave_lock() {
    if LOCKING.load(Ordering::SeqCst) {
        SECTION.leave();
    }
}

/// Holds the critical section while locking is enabled: every layout and creation function takes one.
pub struct Guard(bool);

impl Guard {
    pub fn new() -> Guard {
        if LOCKING.load(Ordering::SeqCst) {
            SECTION.enter();
            Guard(true)
        } else {
            Guard(false)
        }
    }
}

impl Default for Guard {
    fn default() -> Self {
        Guard::new()
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.0 {
            SECTION.leave();
        }
    }
}
