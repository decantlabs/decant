//! A stand-in for a title executable: it imports the Decant shims the way a Scarlett title does
//! (`xmem.dll`, `d3d12_xs.dll`), so the loader must find them before it runs, then calls them and
//! reports through `OutputDebugString`. Built with `cargo build --example stand_in_title`; used to
//! test `decant-launch` without any game file.

#[cfg(windows)]
#[allow(unsafe_code)]
mod imports {
    use std::ffi::c_void;

    #[link(name = "xmem", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void;
        pub fn XMemFree(address: *mut c_void, attributes: u64);
    }

    #[link(name = "d3d12_xs", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn D3D12XboxSetProcessDebugFlags(flags: u32) -> i32;
    }

    #[link(name = "kernel32", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn OutputDebugStringA(text: *const u8);
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn main() {
    use imports::*;
    let say = |text: &str| {
        let line = format!("[stand-in] {text}\n\0");
        // SAFETY: a NUL-terminated string.
        unsafe { OutputDebugStringA(line.as_ptr()) };
    };
    // SAFETY: plain calls into the shims with valid arguments.
    unsafe {
        let p = XMemAlloc(4096, 0);
        say(&format!("XMemAlloc(4096, 0) = {p:p}"));
        if p.is_null() {
            std::process::exit(3);
        }
        p.cast::<u8>().write_bytes(0xAB, 4096);
        XMemFree(p, 0);
        say("XMemFree done");
        let hr = D3D12XboxSetProcessDebugFlags(0);
        say(&format!("D3D12XboxSetProcessDebugFlags(0) = {hr:#010x}"));
    }
}

#[cfg(not(windows))]
fn main() {}
