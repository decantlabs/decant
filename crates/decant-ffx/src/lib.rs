//! `decant_ffx.dll`: forwards the five FidelityFX API calls of a native PC caller to the PC backend
//! named by `DECANT_FFX_BACKEND`. A port of the former `src/decant/ffx_shim.c`, with the same
//! exports and behaviour. The backend must be given by an absolute path (drive or UNC). It is
//! loaded at the first call that finds it, and then stays loaded for the life of the process. A
//! failed load is not remembered, so a later call tries again. Without a backend every call
//! returns 1 (`FFX_API_RETURN_ERROR`).

#[cfg(windows)]
mod exports;

/// What a call returns when no backend answers it.
pub const NO_BACKEND: u32 = 1;

/// Whether a path (UTF-16, without the final zero) may name the backend: at least three units
/// and shorter than the C buffer, and either a UNC path (`\\`) or a drive path (`X:\` or `X:/`).
pub fn backend_path_allowed(path: &[u16]) -> bool {
    let unit = |c: char| c as u16;
    if path.len() < 3 || path.len() >= 32768 {
        return false;
    }
    (path[0] == unit('\\') && path[1] == unit('\\')) || (path[1] == unit(':') && (path[2] == unit('\\') || path[2] == unit('/')))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed(path: &str) -> bool {
        backend_path_allowed(&path.encode_utf16().collect::<Vec<_>>())
    }

    #[test]
    fn only_absolute_paths_name_a_backend() {
        assert!(allowed(r"C:\FidelityFX\amd_fidelityfx_dx12.dll"));
        assert!(allowed("C:/x.dll"));
        assert!(allowed(r"\\server\share\x.dll"));
        assert!(!allowed("amd_fidelityfx_dx12.dll"));
        assert!(!allowed(r"C:x.dll"));
        assert!(!allowed(r".\x.dll"));
        assert!(!allowed(r"\x.dll"));
        assert!(!allowed("C:"));
        assert!(!allowed(""));
        assert!(!allowed(&format!("C:\\{}", "a".repeat(32765))));
        assert!(allowed(&format!("C:\\{}", "a".repeat(32764))));
    }
}
