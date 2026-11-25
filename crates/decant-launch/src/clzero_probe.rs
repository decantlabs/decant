//! Test helper for `tests/cpu_emulation.rs`, not part of the launcher: runs the machine code of a
//! CLZERO loop over a 64 byte aligned buffer filled with 0xAA (with a guard line on each side) and
//! checks that exactly the buffer is zero. First argument `fast` (the exact loop the launcher
//! recognises) or `slow` (a nop inside, so it is emulated per instruction). Exit 0 when right, 8
//! when the buffer is wrong, 9 on a bad argument or no memory.

#[cfg(windows)]
#[allow(unsafe_code)]
fn main() {
    use std::ffi::c_void;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};

    const LEN: usize = 1024;
    #[repr(C, align(64))]
    struct Region([u8; LEN + 128]);

    // mov rax,rcx; lea r10,[rcx+rdx]
    let mut code: Vec<u8> = vec![0x48, 0x89, 0xC8, 0x4C, 0x8D, 0x14, 0x11];
    match std::env::args().nth(1).as_deref() {
        // clzero; add rax,0x40; cmp rax,r10; jb back
        Some("fast") => code.extend([0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4]),
        Some("slow") => code.extend([0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x90, 0x4C, 0x39, 0xD0, 0x72, 0xF3]),
        _ => std::process::exit(9),
    }
    code.push(0xC3);
    // SAFETY: a fresh executable block of the size asked for.
    let memory = unsafe { VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
    if memory.is_null() {
        std::process::exit(9);
    }
    let mut region = Box::new(Region([0xAA; LEN + 128]));
    assert_eq!(region.0.as_ptr() as usize % 64, 0);
    // SAFETY: the block holds at least `code.len()` bytes, and the code follows the Windows x64 call
    // rules: it uses rax, r10, rcx, rdx and writes inside the buffer only.
    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), memory, code.len());
        let run: extern "system" fn(*mut u8, usize) = std::mem::transmute::<*mut c_void, _>(memory.cast());
        run(region.0.as_mut_ptr().add(64), LEN);
    }
    let ok = region.0[..64].iter().all(|&b| b == 0xAA) && region.0[64..64 + LEN].iter().all(|&b| b == 0) && region.0[64 + LEN..].iter().all(|&b| b == 0xAA);
    std::process::exit(if ok { 0 } else { 8 });
}

#[cfg(not(windows))]
fn main() {}
