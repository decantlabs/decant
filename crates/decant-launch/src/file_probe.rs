//! Test helper for `tests/file_calls.rs`, not part of the launcher: calls each file function the debugger
//! redirects for the game drive with paths under the drive letter given as the first argument, the way a
//! title does. The exit code has bit `n` set when call `n` did not do what it should (0: all did), 100 when
//! the argument is not a drive letter. The calls, by bit: see `CALLS` in `tests/file_calls.rs`; what each one
//! changes is checked by that test, in the folder the drive is redirected to.

#![allow(unsafe_code)]

#[cfg(windows)]
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(windows)]
fn main() {
    use std::io::Read;
    use std::os::windows::io::FromRawHandle;
    use windows_sys::Win32::Foundation::{GENERIC_READ, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateDirectoryW, CreateFile2, CreateFileA, CreateFileW, DeleteFileW, FILE_ATTRIBUTE_READONLY, FILE_SHARE_READ, FindClose, FindFirstFileExW, GetFileAttributesA, GetFileAttributesExW, GetFileAttributesW,
        INVALID_FILE_ATTRIBUTES, MoveFileExW, OPEN_EXISTING, RemoveDirectoryW, SetFileAttributesW, WIN32_FILE_ATTRIBUTE_DATA, WIN32_FIND_DATAW,
    };

    let Some(letter) = std::env::args().nth(1).and_then(|a| a.chars().next()).filter(char::is_ascii_alphabetic) else { std::process::exit(100) };
    let read = format!("{letter}:\\pack\\a.dat");
    let ok_content = |handle: *mut std::ffi::c_void| -> bool {
        if handle == INVALID_HANDLE_VALUE {
            return false;
        }
        // SAFETY: `handle` is an open file handle that nothing else owns; the `File` closes it.
        let mut file = unsafe { std::fs::File::from_raw_handle(handle) };
        let mut content = Vec::new();
        file.read_to_end(&mut content).is_ok() && content == b"hello"
    };
    let mut failed = 0u32;
    let mut check = |bit: usize, ok: bool| {
        if !ok {
            failed |= 1 << bit;
        }
    };
    // SAFETY (for the whole block): every pointer is to a live, NUL-terminated buffer of this frame.
    unsafe {
        let w = wide(&read);
        let a = [read.as_bytes(), &[0]].concat();
        check(0, ok_content(CreateFileW(w.as_ptr(), GENERIC_READ, FILE_SHARE_READ, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut())));
        check(1, ok_content(CreateFileA(a.as_ptr(), GENERIC_READ, FILE_SHARE_READ, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut())));
        check(2, ok_content(CreateFile2(w.as_ptr(), GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, std::ptr::null())));
        let long = wide(&format!("\\\\?\\{read}"));
        check(3, ok_content(CreateFileW(long.as_ptr(), GENERIC_READ, FILE_SHARE_READ, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut())));
        check(4, GetFileAttributesW(w.as_ptr()) != INVALID_FILE_ATTRIBUTES);
        check(5, GetFileAttributesA(a.as_ptr()) != INVALID_FILE_ATTRIBUTES);
        let mut data: WIN32_FILE_ATTRIBUTE_DATA = std::mem::zeroed();
        check(6, GetFileAttributesExW(w.as_ptr(), 0, (&raw mut data).cast()) != 0 && data.nFileSizeLow == 5);
        let mut find: WIN32_FIND_DATAW = std::mem::zeroed();
        let handle = FindFirstFileExW(w.as_ptr(), 0, (&raw mut find).cast(), 0, std::ptr::null(), 0);
        check(7, handle != INVALID_HANDLE_VALUE && find.nFileSizeLow == 5);
        if handle != INVALID_HANDLE_VALUE {
            FindClose(handle);
        }
        check(8, CreateDirectoryW(wide(&format!("{letter}:\\pack\\newdir")).as_ptr(), std::ptr::null()) != 0);
        check(9, DeleteFileW(wide(&format!("{letter}:\\pack\\del.txt")).as_ptr()) != 0);
        check(10, RemoveDirectoryW(wide(&format!("{letter}:\\pack\\emptydir")).as_ptr()) != 0);
        check(11, SetFileAttributesW(wide(&format!("{letter}:\\pack\\attr.txt")).as_ptr(), FILE_ATTRIBUTE_READONLY) != 0);
        check(12, MoveFileExW(wide(&format!("{letter}:\\pack\\from.txt")).as_ptr(), wide(&format!("{letter}:\\pack\\to.txt")).as_ptr(), 0) != 0);
        let nt = wide(&format!("\\??\\{read}"));
        check(13, ok_content(CreateFileW(nt.as_ptr(), GENERIC_READ, FILE_SHARE_READ, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut())));
        check(14, CreateDirectoryW(wide(&format!("{letter}:\\newtop")).as_ptr(), std::ptr::null()) != 0);
    }
    std::process::exit(i32::try_from(failed).unwrap_or(i32::MAX));
}

#[cfg(not(windows))]
fn main() {}
