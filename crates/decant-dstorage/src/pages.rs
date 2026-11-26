//! Console requests whose destination is physical pages (option bit 21,
//! `console_options::DESTINATION_PAGES`). The destination is then an array of the title's page frame
//! numbers, as `XMemAllocatePhysicalPages` gave them, one per 64 KiB, and the bytes start at a page
//! offset in the first page. A retail title streams its chunked files this way: it keeps a frame per 64 KiB of a
//! large reservation and passes the part of that table that covers the destination.
//!
//! The engine stages the bytes of such a request (read, then decoded) and hands them to the page pool
//! of `xmem.dll` (`DecantXmemWritePhysicalPages`), which writes them through its own view of the pages.
//! A page the title has not mapped yet gets the bytes all the same, and the pages become dirty for the
//! GPU mirror of title memory. The pool answers where views of the title show the bytes now, which
//! `d3d12_xs.dll` is told, as for a read to an address.
#![allow(unsafe_code)]

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::GetLastError;
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};

use crate::decant::{E_INVALIDARG, E_NOTIMPL, HRESULT, hresult_from_win32};

/// Bytes of one physical page.
pub const PAGE_BYTES: usize = 0x1_0000;

/// The most views a write reports back. A page mapped at more addresses than that is rare, and the
/// ones past it only miss the note to `d3d12_xs.dll`.
const VIEWS: usize = 64;

/// Where the bytes of page destinations go.
pub trait PagePool: Send + Sync {
    /// Writes `data` from byte `offset` of the page of `frames[0]` on, one frame per 64 KiB. All or
    /// nothing. Returns where views of the title show the bytes now, as `(address, bytes)`.
    fn write(&self, frames: &[u64], offset: usize, data: &[u8]) -> Result<Vec<(usize, usize)>, HRESULT>;
}

/// The page pool of `xmem.dll`, found in the process by name at each write.
pub struct XmemPool;

type WritePhysicalPages = unsafe extern "system" fn(*const u64, usize, usize, *const u8, usize, *mut [usize; 2], u32, *mut u32) -> i32;

fn resolve_write_physical_pages() -> Option<WritePhysicalPages> {
    static EXPORT: OnceLock<WritePhysicalPages> = OnceLock::new();
    if let Some(&f) = EXPORT.get() {
        return Some(f);
    }
    // SAFETY: lookups by name in a module that is loaded or not.
    let export = unsafe {
        let module = GetModuleHandleW(windows_sys::w!("xmem.dll"));
        if module.is_null() { None } else { GetProcAddress(module, c"DecantXmemWritePhysicalPages".as_ptr().cast()) }
    }?;
    // SAFETY: the export has this signature.
    let f: WritePhysicalPages = unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, WritePhysicalPages>(export) };
    let _ = EXPORT.set(f);
    Some(f)
}

impl PagePool for XmemPool {
    fn write(&self, frames: &[u64], offset: usize, data: &[u8]) -> Result<Vec<(usize, usize)>, HRESULT> {
        let Some(write) = resolve_write_physical_pages() else {
            decant_log::calls::not_implemented("DirectStorage page destination without DecantXmemWritePhysicalPages in xmem.dll");
            return Err(E_NOTIMPL);
        };
        let mut views = [[0usize; 2]; VIEWS];
        let mut found = 0u32;
        // SAFETY: `frames` and `data` are readable for their lengths, `views` has room for `VIEWS` pairs.
        let ok = unsafe { write(frames.as_ptr(), frames.len(), offset, data.as_ptr(), data.len(), views.as_mut_ptr(), VIEWS as u32, &mut found) };
        if ok == 0 {
            // SAFETY: no precondition.
            return Err(hresult_from_win32(unsafe { GetLastError() }));
        }
        Ok(views.iter().take((found as usize).min(VIEWS)).map(|v| (v[0], v[1])).collect())
    }
}

/// The page destination of a request: the frames, read from the title's array at enqueue (a title
/// may pass an array on its stack), where the bytes start in the first page, and the array itself.
///
/// The title enqueues a read before it has put the frames into its array: at enqueue every entry is 0
/// (no frame, measured on every page destination of a title run), and the frames are there
/// by the time the read runs. So a snapshot with a 0 entry is taken again from the array when the
/// bytes are written ([`PageTarget::frames_now`]).
pub struct PageTarget {
    pub frames: Vec<u64>,
    pub offset: usize,
    array: usize,
}

/// Frame 0 is no frame: `xmem.dll` never hands it out.
const NO_FRAME: u64 = 0;

impl PageTarget {
    /// The frames to write to now, borrowing the snapshot when every entry is already resolved.
    pub fn frames_now_cow(&self) -> Cow<'_, [u64]> {
        if !self.frames.contains(&NO_FRAME) {
            return Cow::Borrowed(&self.frames);
        }
        let bytes = self.frames.len() * 8;
        if !readable(self.array, bytes) {
            return Cow::Borrowed(&self.frames);
        }
        let array = self.array as *const u64;
        // SAFETY: the range is committed readable memory (checked just above), and the array may be unaligned.
        Cow::Owned((0..self.frames.len()).map(|k| unsafe { array.add(k).read_unaligned() }).collect())
    }

    /// The frames to write to now: the snapshot of the enqueue, or, when it holds no frame for a
    /// page, the title's array as it is now, if that memory is still readable. Then the snapshot is
    /// what the write refuses.
    #[cfg(test)]
    pub fn frames_now(&self) -> Vec<u64> {
        self.frames_now_cow().into_owned()
    }

    /// The frames a destination of `size` bytes from `offset` (the low 16 bits of the page offset
    /// field) covers, read from `array`. `E_INVALIDARG` for a null array.
    ///
    /// # Safety
    /// `array` is null or readable for the entries a destination of `size` bytes from `offset` needs.
    pub unsafe fn read(array: *const c_void, offset_field: u64, size: u32) -> Result<PageTarget, HRESULT> {
        if array.is_null() {
            return Err(E_INVALIDARG);
        }
        let offset = (offset_field & 0xFFFF) as usize;
        let count = (offset + size as usize).div_ceil(PAGE_BYTES);
        // SAFETY: the caller guarantees `count` entries, and the array may be unaligned.
        let frames = (0..count).map(|k| unsafe { array.cast::<u64>().add(k).read_unaligned() }).collect();
        Ok(PageTarget { frames, offset, array: array as usize })
    }
}

/// Whether `[address, address + len)` is committed memory that can be read.
fn readable(address: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let end = address.saturating_add(len);
    let mut at = address;
    while at < end {
        // SAFETY: a query of this process's address space into a local.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        if info.State != MEM_COMMIT || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 || info.Protect == 0 {
            return false;
        }
        at = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_the_title_puts_in_after_the_enqueue_are_taken_at_the_write() {
        let mut array: Vec<u64> = vec![0, 0];
        // SAFETY: the array holds the two entries a destination of two pages needs.
        let target = unsafe { PageTarget::read(array.as_ptr().cast(), 0, 2 * PAGE_BYTES as u32).unwrap() };
        assert_eq!(target.frames, [0, 0]);
        array[0] = 160;
        array[1] = 176;
        assert_eq!(target.frames_now(), [160, 176], "the array as it is when the bytes are written");
    }

    #[test]
    fn a_full_snapshot_is_kept_whatever_the_array_holds_later() {
        let mut array: Vec<u64> = vec![16, 32];
        // SAFETY: as above.
        let target = unsafe { PageTarget::read(array.as_ptr().cast(), 0, 2 * PAGE_BYTES as u32).unwrap() };
        array[0] = 0;
        array[1] = 999;
        assert_eq!(target.frames_now(), [16, 32], "a stack array may hold other data by then");
    }

    #[test]
    fn an_array_that_is_gone_leaves_the_snapshot() {
        let target = PageTarget { frames: vec![0], offset: 0, array: 0x10 };
        assert_eq!(target.frames_now(), [0], "address 0x10 is never readable");
    }

    #[test]
    fn the_target_covers_the_pages_the_bytes_reach_from_the_offset() {
        let array: [u64; 4] = [16, 48, 32, 80];
        // SAFETY: the array holds four entries, more than any of these reads needs.
        unsafe {
            let one = PageTarget::read(array.as_ptr().cast(), 0, PAGE_BYTES as u32).unwrap();
            assert_eq!((one.frames, one.offset), (vec![16], 0));
            let two = PageTarget::read(array.as_ptr().cast(), 0xABCD_0010, PAGE_BYTES as u32).unwrap();
            assert_eq!((two.frames, two.offset), (vec![16, 48], 0x10), "only the low 16 bits are the offset");
            let middle = PageTarget::read(array.as_ptr().cast(), 0x100, 0x4000).unwrap();
            assert_eq!(middle.frames, [16], "a destination that ends in the middle of its page");
            let three = PageTarget::read(array.as_ptr().cast(), 0xFFFF, PAGE_BYTES as u32 + 2).unwrap();
            assert_eq!(three.frames, [16, 48, 32]);
            assert_eq!(PageTarget::read(std::ptr::null(), 0, 1).err(), Some(E_INVALIDARG));
        }
    }

    #[test]
    fn frames_now_cow_borrows_when_all_frames_are_resolved() {
        let array: [u64; 2] = [16, 32];
        // SAFETY: the array holds two entries.
        let target = unsafe { PageTarget::read(array.as_ptr().cast(), 0, 2 * PAGE_BYTES as u32).unwrap() };
        assert!(matches!(target.frames_now_cow(), Cow::Borrowed(_)));
        assert_eq!(target.frames_now_cow().as_ref(), &[16, 32]);

        let mut deferred = [0u64, 0u64];
        // SAFETY: the array holds two entries.
        let target_deferred = unsafe { PageTarget::read(deferred.as_ptr().cast(), 0, 2 * PAGE_BYTES as u32).unwrap() };
        deferred[0] = 64;
        deferred[1] = 128;
        std::hint::black_box(&deferred);
        assert!(matches!(target_deferred.frames_now_cow(), Cow::Owned(_)));
        assert_eq!(target_deferred.frames_now_cow().as_ref(), &[64, 128]);
    }
}
