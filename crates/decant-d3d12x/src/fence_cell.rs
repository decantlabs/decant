//! Memory cells that mirror the completed value of a PC fence.
//!
//! Slot 12 of the console fence (newer layout) returns the address of the fence's completed value,
//! and a retail title polls that 64 bit value with MONITORX and MWAITX instead of calling
//! `GetCompletedValue`. A PC fence keeps its value private, so each fence that is asked for its
//! address gets a cell of our own. One watcher thread copies the PC value into the cells whenever
//! a signal that is still pending completes.
//!
//! The queue signals tell this module their target value (`queued`). Once a fence has a cell, the
//! watcher asks for a wake up at that value through one shared auto reset event. Signals made by
//! the CPU refresh the cells at once.

#![allow(unsafe_code)]

use crate::log::xs_log;
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

type Object = *mut c_void;

/// `ID3D12Fence::GetCompletedValue`, vtable slot 8.
type GetCompletedValue = unsafe extern "system" fn(Object) -> u64;
/// `ID3D12Fence::SetEventOnCompletion`, vtable slot 9.
type SetEventOnCompletion = unsafe extern "system" fn(Object, u64, *mut c_void) -> i32;

const GET_COMPLETED_VALUE_SLOT: usize = 8;
const SET_EVENT_ON_COMPLETION_SLOT: usize = 9;

/// A completed value on a 64 byte line of its own, so MONITORX on it wakes for that value only.
#[repr(C, align(64))]
struct Cell(AtomicU64);

/// What is known of a fence: the cell it got (once its address was asked for) and the highest value
/// a queue was told to signal.
#[derive(Default)]
struct Known {
    cell: Option<&'static Cell>,
    highest_queued: u64,
    /// The last value the title was given (rules R1 and R2): never lower, never above `MAX_VALUE`.
    last: u64,
}

/// The largest value a fence may show the title: 56 bits, the range of the title's modular
/// comparison (rule R2).
pub const MAX_VALUE: u64 = (1 << 56) - 1;

/// What one PC read gives the title (rules R1 and R2).
#[derive(Debug, PartialEq, Eq)]
pub enum Reading {
    /// A valid value, not below the last one: it becomes the last valid value.
    Value(u64),
    /// `u64::MAX` on a removed device: the title's lost-device answer.
    Lost,
    /// A read the layer does not pass on: the last valid value is given instead.
    Refused { read: u64, kept: u64 },
}

/// Applies R1 and R2 to one PC read. `removed`: `GetDeviceRemovedReason` is not S_OK. `kept`: the
/// last valid value given to the title. Pure, for the tests.
pub fn reading(read: u64, removed: bool, kept: u64) -> Reading {
    if read == u64::MAX {
        return if removed { Reading::Lost } else { Reading::Refused { read, kept } };
    }
    if read > MAX_VALUE || read < kept {
        return Reading::Refused { read, kept };
    }
    Reading::Value(read)
}

/// Whether the device of PC fence `fence` is removed (`GetDeviceRemovedReason` is not S_OK).
#[cfg(windows)]
fn device_removed(fence: usize) -> bool {
    type GetDevice = unsafe extern "system" fn(Object, *const [u8; 16], *mut Object) -> i32;
    type GetRemoved = unsafe extern "system" fn(Object) -> i32;
    type Release = unsafe extern "system" fn(Object) -> u32;
    // IID_ID3D12Device, {189819f1-1db6-4b57-be54-1821339b85f7}, in memory order.
    const IID_ID3D12DEVICE: [u8; 16] = [0xf1, 0x19, 0x98, 0x18, 0xb6, 0x1d, 0x57, 0x4b, 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7];
    let mut device: Object = std::ptr::null_mut();
    // SAFETY: `fence` is a live PC fence (only those are in the table); the device is released below.
    unsafe {
        let get: GetDevice = method(fence, 7);
        if get(fence as Object, &IID_ID3D12DEVICE, &mut device) < 0 || device.is_null() {
            return false;
        }
        let reason: GetRemoved = method(device as usize, 37);
        let removed = reason(device) != 0;
        let release: Release = method(device as usize, 2);
        release(device);
        removed
    }
}

#[cfg(not(windows))]
fn device_removed(_fence: usize) -> bool {
    false
}

/// The value the title is given for PC fence `fence`, applying [`reading`] and keeping the last one.
fn settle(fence: usize, k: &mut Known) -> u64 {
    // SAFETY: only live PC fences are ever in the table.
    let read = unsafe { method::<GetCompletedValue>(fence, GET_COMPLETED_VALUE_SLOT)(fence as Object) };
    let removed = read == u64::MAX && device_removed(fence);
    match reading(read, removed, k.last) {
        Reading::Value(v) => {
            k.last = v;
            v
        }
        Reading::Lost => u64::MAX,
        Reading::Refused { read, kept } => {
            xs_log!("fence cells: PC fence {fence:#x} reads {read:#x} on a live device, or outside 56 bits, or below {kept:#x}: the title is given {kept:#x}");
            kept
        }
    }
}

#[derive(Default)]
struct FastUsizeHasher(u64);

impl Hasher for FastUsizeHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(5) ^ u64::from(b);
        }
    }

    fn write_usize(&mut self, i: usize) {
        self.0 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<V> = HashMap<usize, V, BuildHasherDefault<FastUsizeHasher>>;

/// All fences seen, by address. A fence stays here for the life of the process: the fence proxy
/// holds a reference and the title never frees a fence it polls.
static KNOWN: Mutex<Option<FastMap<Known>>> = Mutex::new(None);

fn known() -> MutexGuard<'static, Option<FastMap<Known>>> {
    KNOWN.lock().unwrap_or_else(|e| e.into_inner())
}

/// The function pointer in vtable slot `slot` of `fence`.
///
/// # Safety
/// `fence` is a live PC `ID3D12Fence` and `F` is the type of the method in that slot.
unsafe fn method<F: Copy>(fence: usize, slot: usize) -> F {
    // SAFETY: a COM object starts with a pointer to its vtable.
    unsafe {
        let vtable = *(fence as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*vtable.add(slot))
    }
}

/// The event the PC fences set, as a raw handle value. The first call makes it together with the
/// watcher thread that waits on it. Without a Windows host there is no event and the value is 0.
#[cfg(windows)]
fn watcher_event() -> usize {
    use std::sync::OnceLock;
    use windows_sys::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    static EVENT: OnceLock<usize> = OnceLock::new();
    *EVENT.get_or_init(|| {
        // SAFETY: an unnamed auto reset event that starts unsignalled.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) } as usize;
        decant_log::handles::created(event, "fence cell wake event");
        let started = std::thread::Builder::new().name("d3d12xs-fence-cells".into()).spawn(move || {
            loop {
                // SAFETY: the event is never closed.
                unsafe { WaitForSingleObject(event as *mut c_void, INFINITE) };
                copy_values();
            }
        });
        if let Err(e) = started {
            xs_log!("fence cells: no watcher thread ({e}), slot 12 values only move on CPU signals");
        }
        event
    })
}

#[cfg(not(windows))]
fn watcher_event() -> usize {
    0
}

/// Stores the completed value of every fence that has a cell into that cell.
fn copy_values() {
    let mut guard = known();
    for (&fence, k) in guard.iter_mut().flatten() {
        let Some(cell) = k.cell else { continue };
        let value = settle(fence, k);
        cell.0.store(value, Ordering::Release);
    }
}

/// `GetCompletedValue` as the title is given it (slot 8): the same answer the cell holds (R1, R2).
pub fn completed_value(fence: Object) -> u64 {
    if fence.is_null() {
        return 0;
    }
    let mut guard = known();
    let k = guard.get_or_insert_with(FastMap::default).entry(fence as usize).or_default();
    let value = settle(fence as usize, k);
    if let Some(cell) = k.cell {
        cell.0.store(value, Ordering::Release);
    }
    value
}

/// Has `fence` set the watcher's event when it reaches `value`.
fn wake_at(fence: usize, value: u64) {
    let event = watcher_event();
    if event == 0 {
        return;
    }
    // SAFETY: `fence` is a live PC fence and the event handle stays valid.
    let hr = unsafe { method::<SetEventOnCompletion>(fence, SET_EVENT_ON_COMPLETION_SLOT)(fence as Object, value, event as *mut c_void) };
    if hr < 0 {
        xs_log!("fence cells: SetEventOnCompletion({value:#x}) on {fence:#x} failed, hr {hr:#x}");
    }
}

/// Whether `queued` needs to register a wake-up event for `value` given the fence's `has_cell` and `last` known value.
fn should_wake(has_cell: bool, value: u64, last: u64) -> bool {
    has_cell && value > last
}

/// A queue was told to signal `fence` up to `value`.
pub fn queued(fence: Object, value: u64) {
    if fence.is_null() {
        return;
    }
    let wake = {
        let mut guard = known();
        let k = guard.get_or_insert_with(FastMap::default).entry(fence as usize).or_default();
        k.highest_queued = k.highest_queued.max(value);
        should_wake(k.cell.is_some(), value, k.last)
    };
    if wake {
        wake_at(fence as usize, value);
    }
}

/// The CPU signalled `fence` to `value`: the highest value known moves with it, then its cell follows.
pub fn cpu_signalled(fence: Object, value: u64) {
    if fence.is_null() {
        return;
    }
    let mut guard = known();
    let k = guard.get_or_insert_with(FastMap::default).entry(fence as usize).or_default();
    k.highest_queued = k.highest_queued.max(value);
    if let Some(cell) = k.cell {
        let settled = settle(fence as usize, k);
        cell.0.store(settled, Ordering::Release);
    }
}

/// The address of the cell with `fence`'s completed value. The first call for a fence makes the cell.
pub fn address(fence: Object) -> *const u64 {
    let (cell, queued_before) = {
        let mut guard = known();
        let k = guard.get_or_insert_with(FastMap::default).entry(fence as usize).or_default();
        if let Some(cell) = k.cell {
            return cell.0.as_ptr();
        }
        let cell: &'static Cell = Box::leak(Box::new(Cell(AtomicU64::new(0))));
        k.cell = Some(cell);
        (cell, k.highest_queued)
    };
    if queued_before > 0 {
        wake_at(fence as usize, queued_before);
    }
    copy_values();
    cell.0.as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cell_owns_its_cache_line() {
        assert_eq!(align_of::<Cell>(), 64);
        assert_eq!(size_of::<Cell>(), 64);
    }

    #[test]
    fn a_null_fence_is_ignored() {
        queued(std::ptr::null_mut(), 5);
        cpu_signalled(std::ptr::null_mut(), 5);
        assert_eq!(completed_value(std::ptr::null_mut()), 0);
        assert!(known().as_ref().is_none_or(|t| !t.contains_key(&0)));
    }

    #[test]
    fn queued_skips_wake_when_value_is_already_reached() {
        assert!(!should_wake(false, 10, 5));
        assert!(!should_wake(true, 5, 5));
        assert!(!should_wake(true, 4, 5));
        assert!(should_wake(true, 6, 5));
    }
}

#[cfg(test)]
mod reading_tests {
    use super::*;

    /// rule R6, first direction: MAX on a live device is not given; the last valid value is.
    #[test]
    fn max_on_a_live_device_is_refused_and_the_last_valid_value_is_kept() {
        assert_eq!(reading(u64::MAX, false, 41), Reading::Refused { read: u64::MAX, kept: 41 });
    }

    /// rule R6, second direction: with the device removed, MAX is given.
    #[test]
    fn max_with_a_removed_device_is_given() {
        assert_eq!(reading(u64::MAX, true, 41), Reading::Lost);
    }

    /// R2: up to 56 bits pass; above does not; the value never goes back.
    #[test]
    fn values_stay_within_56_bits_and_never_go_back() {
        assert_eq!(reading(MAX_VALUE, false, 9), Reading::Value(MAX_VALUE));
        assert_eq!(reading(1 << 56, false, 9), Reading::Refused { read: 1 << 56, kept: 9 });
        assert_eq!(reading(7, false, 8), Reading::Refused { read: 7, kept: 8 });
        assert_eq!(reading(8, false, 8), Reading::Value(8));
    }
}
