//! The five exports and the backend loader. The arguments are the caller's FidelityFX
//! structures, passed through untouched, so the forwarding is `unsafe` by nature.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW};

use crate::{NO_BACKEND, backend_path_allowed};

/// The loaded backend, 0 until a load succeeds.
static BACKEND: AtomicUsize = AtomicUsize::new(0);
/// Serialises the loading attempts, as `InitOnceExecuteOnce` did.
static LOADING: Mutex<()> = Mutex::new(());

fn this_module() -> HMODULE {
    unsafe extern "C" {
        static __ImageBase: u8;
    }
    // Only the address of the linker defined symbol is taken, it is never read.
    (&raw const __ImageBase) as HMODULE
}

/// Loads the backend named by `DECANT_FFX_BACKEND`, or returns 0.
fn load_backend() -> usize {
    let Some(path) = std::env::var_os("DECANT_FFX_BACKEND") else {
        return 0;
    };
    let mut wide: Vec<u16> = path.encode_wide().collect();
    if !backend_path_allowed(&wide) {
        return 0;
    }
    wide.push(0);
    // SAFETY: `wide` is a zero terminated absolute path. The search is limited to the DLL's own
    // folder and System32 for its dependencies.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32) };
    if !module.is_null() && module == this_module() {
        // The variable names this shim itself: refuse it, it would forward to itself.
        // SAFETY: `module` holds the reference LoadLibraryExW just added.
        unsafe { FreeLibrary(module) };
        return 0;
    }
    module as usize
}

fn backend() -> usize {
    let loaded = BACKEND.load(Ordering::Acquire);
    if loaded != 0 {
        return loaded;
    }
    let _guard = LOADING.lock().unwrap_or_else(|e| e.into_inner());
    let loaded = BACKEND.load(Ordering::Acquire);
    if loaded != 0 {
        return loaded;
    }
    let module = load_backend();
    BACKEND.store(module, Ordering::Release);
    module
}

/// The backend's export `name` (zero terminated), or null.
fn resolve(name: &[u8]) -> *const c_void {
    let module = backend();
    if module == 0 {
        return std::ptr::null();
    }
    // SAFETY: the module stays loaded for the life of the process and `name` ends with a zero.
    unsafe { GetProcAddress(module as HMODULE, name.as_ptr()) }.map_or(std::ptr::null(), |f| f as *const c_void)
}

type CreateFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_void) -> u32;
type ConstArgFn = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
type MutArgFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> u32;

/// # Safety
/// Exported. The arguments go to the backend's `ffxCreateContext` unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffxCreateContext(context: *mut *mut c_void, argument: *mut c_void, allocator: *const c_void) -> u32 {
    let f = resolve(b"ffxCreateContext\0");
    // SAFETY: the backend exports ffxCreateContext with this signature (ffx_api.h).
    if f.is_null() { NO_BACKEND } else { unsafe { std::mem::transmute::<*const c_void, CreateFn>(f)(context, argument, allocator) } }
}

/// # Safety
/// Exported. The arguments go to the backend's `ffxDestroyContext` unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffxDestroyContext(context: *mut *mut c_void, argument: *const c_void) -> u32 {
    let f = resolve(b"ffxDestroyContext\0");
    // SAFETY: the backend exports ffxDestroyContext with this signature (ffx_api.h).
    if f.is_null() { NO_BACKEND } else { unsafe { std::mem::transmute::<*const c_void, ConstArgFn>(f)(context, argument) } }
}

/// # Safety
/// Exported. The arguments go to the backend's `ffxConfigure` unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffxConfigure(context: *mut *mut c_void, argument: *const c_void) -> u32 {
    let f = resolve(b"ffxConfigure\0");
    // SAFETY: the backend exports ffxConfigure with this signature (ffx_api.h).
    if f.is_null() { NO_BACKEND } else { unsafe { std::mem::transmute::<*const c_void, ConstArgFn>(f)(context, argument) } }
}

/// # Safety
/// Exported. The arguments go to the backend's `ffxQuery` unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffxQuery(context: *mut *mut c_void, argument: *mut c_void) -> u32 {
    let f = resolve(b"ffxQuery\0");
    // SAFETY: the backend exports ffxQuery with this signature (ffx_api.h).
    if f.is_null() { NO_BACKEND } else { unsafe { std::mem::transmute::<*const c_void, MutArgFn>(f)(context, argument) } }
}

/// # Safety
/// Exported. The arguments go to the backend's `ffxDispatch` unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffxDispatch(context: *mut *mut c_void, argument: *const c_void) -> u32 {
    let f = resolve(b"ffxDispatch\0");
    // SAFETY: the backend exports ffxDispatch with this signature (ffx_api.h).
    if f.is_null() { NO_BACKEND } else { unsafe { std::mem::transmute::<*const c_void, ConstArgFn>(f)(context, argument) } }
}
