//! Shared by the integration tests: a [`Compiler`] over the DXC the layer loads (`decant-dxc`),
//! so a test compiles a translation with the same `dxcompiler.dll` and the same arguments as
//! `d3d12_xs.dll` does in a run.
//!
//! The folder of `dxcompiler.dll` and `dxil.dll` is searched in the environment variable
//! `DECANT_DXC_DIR`, then beside the test program, then in `Windows Kits\10\bin\<version>\x64`,
//! newest version first. Tests that need it skip with a message when it is missing, so the suite
//! runs on every machine.
#![allow(dead_code)]

use std::sync::OnceLock;

use decant_shader::api::Compiler;

/// The DXC of this test program, loaded once.
pub struct Dxc(&'static decant_dxc::Dxc);

pub fn find_dxc() -> Option<Dxc> {
    static LOADED: OnceLock<Option<decant_dxc::Dxc>> = OnceLock::new();
    LOADED
        .get_or_init(|| match decant_dxc::Dxc::load() {
            Ok(dxc) => Some(dxc),
            Err(why) => {
                eprintln!("no DXC: {why}");
                None
            }
        })
        .as_ref()
        .map(Dxc)
}

impl Compiler for Dxc {
    fn compile(
        &self,
        hlsl: &str,
        entry: &str,
        profile: &str,
        args: &[String],
    ) -> Result<Vec<u8>, String> {
        self.0.compile(hlsl, entry, profile, args)
    }
}
