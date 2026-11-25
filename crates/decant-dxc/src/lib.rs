//! The DXC host of the direct shader path: loads `dxcompiler.dll` and `dxil.dll` at
//! run time and compiles HLSL to signed DXIL through `IDxcCompiler3`. One loader for the layer
//! (`d3d12_xs.dll`) and for the tests of the translator (`decant-shader`), so both compile with
//! the same DXC and the same arguments.
//!
//! Nothing but `LoadLibrary`, `GetProcAddress` and the COM vtables of the public `dxcapi.h` is used,
//! so the same code runs under Wine and Proton with the Windows DXC DLLs. On another host the crate
//! builds and [`Dxc::load`] says that DXC is missing.
//!
//! Search order for the folder that holds `dxcompiler.dll`:
//! 1. `DECANT_DXC_DIR`,
//! 2. the folder of the module that holds this code (`d3d12_xs.dll`, or the test program),
//! 3. the Windows SDK bin folders (`Windows Kits\10\bin\<version>\x64`), newest first,
//! 4. the loader's own search (the title's folder, the system folders, `PATH`).
//!
//! `dxil.dll` (the validator that signs the DXIL) is loaded from the same folder first, so that
//! `dxcompiler.dll` finds it when it asks for it by name. Without it DXC still compiles, but the
//! DXIL is not signed and a Windows D3D12 runtime refuses it; [`signed`] tells.
//!
//! Public references: the DirectXShaderCompiler `dxcapi.h` and the "Using dxc.exe and dxcompiler.dll"
//! page of its wiki.

/// Whether a DXIL container is signed: its 16 byte hash after the magic is not all zero.
pub fn signed(container: &[u8]) -> bool {
    container.len() >= 20 && &container[..4] == b"DXBC" && container[4..20].iter().any(|b| *b != 0)
}

#[cfg(windows)]
mod host;
#[cfg(windows)]
pub use host::{Dxc, candidate_dirs, sdk_dirs};

#[cfg(not(windows))]
mod absent {
    use std::path::{Path, PathBuf};

    /// DXC on a host that cannot load the Windows DLLs: never loads.
    pub struct Dxc {
        /// Where `dxcompiler.dll` came from.
        pub dir: PathBuf,
        /// Whether `dxil.dll` was found beside it.
        pub validator: bool,
    }

    impl Dxc {
        /// Always the error that DXC needs Windows (or Wine).
        pub fn load() -> Result<Dxc, String> {
            Err("dxcompiler.dll is loaded on Windows only".to_string())
        }

        /// As [`Dxc::load`].
        pub fn load_from(_dir: Option<&Path>) -> Result<Dxc, String> {
            Dxc::load()
        }

        /// Never reached: there is no value of this type.
        pub fn describe(&self) -> String {
            String::new()
        }

        /// Never reached.
        pub fn version(&self) -> String {
            String::new()
        }

        /// Never reached.
        pub fn compile(&self, _hlsl: &str, _entry: &str, _profile: &str, _extra: &[String]) -> Result<Vec<u8>, String> {
            Err("no DXC".to_string())
        }
    }
}
#[cfg(not(windows))]
pub use absent::Dxc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_signed_check_reads_the_container_hash() {
        let mut c = b"DXBC".to_vec();
        c.extend_from_slice(&[0; 16]);
        c.extend_from_slice(&[1, 0, 0, 0]);
        assert!(!signed(&c));
        c[7] = 9;
        assert!(signed(&c));
        assert!(!signed(b"DXBC"));
    }
}
