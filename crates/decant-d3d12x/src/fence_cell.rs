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
}

/// All fences seen, by address. A fence stays here for the life of the process: the fence proxy
/// holds a reference and the title never frees a fence it polls.
static KNOWN: Mutex<Option<HashMap<usize, Known>>> = Mutex::new(None);

fn known() -> MutexGuard<'static, Option<HashMap<usize, Known>>> {
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
    let guard = known();
    for (&fence, k) in guard.iter().flatten() {
        let Some(cell) = k.cell else { continue };
        // SAFETY: only live PC fences are ever in the table.
        let value = unsafe { method::<GetCompletedValue>(fence, GET_COMPLETED_VALUE_SLOT)(fence as Object) };
        cell.0.store(value, Ordering::Release);
    }
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

/// A queue was told to signal `fence` up to `value`.
pub fn queued(fence: Object, value: u64) {
    if fence.is_null() {
        return;
    }
    let has_cell = {
        let mut guard = known();
        let k = guard.get_or_insert_with(HashMap::new).entry(fence as usize).or_default();
        k.highest_queued = k.highest_queued.max(value);
        k.cell.is_some()
    };
    if has_cell {
        wake_at(fence as usize, value);
    }
}

/// The CPU signalled `fence`.
pub fn cpu_signalled(fence: Object) {
    let has_cell = known().as_ref().and_then(|t| t.get(&(fence as usize))).is_some_and(|k| k.cell.is_some());
    if has_cell {
        copy_values();
    }
}

/// The address of the cell with `fence`'s completed value. The first call for a fence makes the cell.
pub fn address(fence: Object) -> *const u64 {
    let (cell, queued_before) = {
        let mut guard = known();
        let k = guard.get_or_insert_with(HashMap::new).entry(fence as usize).or_default();
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
        assert!(known().as_ref().is_none_or(|t| !t.contains_key(&0)));
    }
}
