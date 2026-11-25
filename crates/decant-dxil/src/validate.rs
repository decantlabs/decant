//! Test helper (Windows only): validates a container in place through `IDxcValidator` of
//! `dxil.dll`, the library of the Windows SDK or of a DXC release (specification section 5.1).
//!
//! `dxil.dll` is searched in the environment variable `DECANT_SHADER_DXIL_DLL`, then in
//! `Windows Kits\10\bin\<version>\x64` (newest version first), then through the normal DLL search path.
#![allow(unsafe_code)]

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D::Dxc::{CLSID_DxcValidator, DxcValidatorFlags_InPlaceEdit, IDxcBlob, IDxcBlob_Impl, IDxcValidator};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{GUID, HRESULT, Interface, PCSTR, PCWSTR};
use windows_core::implement;

/// Why a validation did not succeed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidateError {
    /// `dxil.dll` or its validator could not be loaded. Tests skip in that case.
    Missing(String),
    /// The validator rejected the container: its status and message.
    Rejected(String),
}

impl std::fmt::Display for ValidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidateError::Missing(m) => write!(f, "validator not available: {m}"),
            ValidateError::Rejected(m) => write!(f, "validator rejected the container: {m}"),
        }
    }
}

/// The candidate paths of `dxil.dll`, best first.
pub fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("DECANT_SHADER_DXIL_DLL") {
        out.push(PathBuf::from(p));
    }
    for root in ["C:\\Program Files (x86)\\Windows Kits\\10\\bin", "C:\\Program Files\\Windows Kits\\10\\bin"] {
        let Ok(dir) = std::fs::read_dir(root) else { continue };
        let mut versions: Vec<PathBuf> = dir.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("x64").join("dxil.dll").is_file()).collect();
        versions.sort();
        out.extend(versions.into_iter().rev().map(|p| p.join("x64").join("dxil.dll")));
    }
    out.push(PathBuf::from("dxil.dll"));
    out
}

/// Where `dxil.dll` was found, if anywhere.
pub fn find_dxil() -> Option<PathBuf> {
    candidates().into_iter().find(|p| p.components().count() == 1 || p.is_file())
}

#[implement(IDxcBlob)]
struct Blob(UnsafeCell<Vec<u8>>);

impl IDxcBlob_Impl for Blob_Impl {
    fn GetBufferPointer(&self) -> *mut c_void {
        // SAFETY: the vector is only reached through this object and the validator, one at a time.
        unsafe { (*self.0.get()).as_mut_ptr().cast() }
    }
    fn GetBufferSize(&self) -> usize {
        // SAFETY: as above.
        unsafe { (*self.0.get()).len() }
    }
}

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

fn create_validator() -> Result<IDxcValidator, ValidateError> {
    let path = find_dxil().ok_or_else(|| ValidateError::Missing("dxil.dll not found (set DECANT_SHADER_DXIL_DLL)".into()))?;
    let wide: Vec<u16> = path.as_os_str().to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: the DLL stays loaded for the life of the process, the entry point has the documented signature.
    unsafe {
        let module = LoadLibraryW(PCWSTR(wide.as_ptr())).map_err(|e| ValidateError::Missing(format!("{}: {e}", path.display())))?;
        let proc = GetProcAddress(module, PCSTR(c"DxcCreateInstance".as_ptr().cast())).ok_or_else(|| ValidateError::Missing("DxcCreateInstance is not exported".into()))?;
        let create: CreateInstance = std::mem::transmute(proc);
        let mut out: *mut c_void = std::ptr::null_mut();
        let hr = create(&CLSID_DxcValidator, &IDxcValidator::IID, &mut out);
        if hr.is_err() || out.is_null() {
            return Err(ValidateError::Missing(format!("DxcCreateInstance(CLSID_DxcValidator) failed with {:#010x}", hr.0)));
        }
        Ok(IDxcValidator::from_raw(out))
    }
}

/// Validates `container` with `DxcValidatorFlags_InPlaceEdit`. The validator writes the container
/// digest it computes into the bytes of `container` (and may touch other bytes it owns), so the
/// slice is updated.
pub fn validate_in_place(container: &mut [u8]) -> Result<(), ValidateError> {
    let validator = create_validator()?;
    let blob: IDxcBlob = Blob(UnsafeCell::new(container.to_vec())).into();
    // SAFETY: the blob is a valid COM object that outlives the call.
    unsafe {
        let result = validator.Validate(&blob, DxcValidatorFlags_InPlaceEdit).map_err(|e| ValidateError::Rejected(format!("Validate failed: {e}")))?;
        let status = result.GetStatus().map_err(|e| ValidateError::Rejected(format!("GetStatus failed: {e}")))?;
        if status.is_err() {
            let text = match result.GetErrorBuffer() {
                Ok(buf) => {
                    let b: IDxcBlob = buf.cast().expect("an encoding blob is a blob");
                    let (p, n) = (b.GetBufferPointer(), b.GetBufferSize());
                    if p.is_null() { String::new() } else { String::from_utf8_lossy(std::slice::from_raw_parts(p.cast::<u8>(), n)).into_owned() }
                }
                Err(_) => String::new(),
            };
            return Err(ValidateError::Rejected(format!("status {:#010x}: {}", status.0, text.trim())));
        }
        let (p, n) = (blob.GetBufferPointer(), blob.GetBufferSize());
        if n == container.len() {
            container.copy_from_slice(std::slice::from_raw_parts(p.cast::<u8>(), n));
        }
    }
    Ok(())
}
