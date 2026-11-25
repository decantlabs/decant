//! Link settings of `xgameruntime.dll` and of its test program.
//!
//! The DLL must export the seven entry points of the PC runtime under the PC runtime's ordinals,
//! as the former `xgameruntime.def` did. `DllCanUnloadNow` is a private COM export with a linker
//! chosen ordinal (it is never imported by ordinal).
//!
//! When `DECANT_GDK_ROOT` names the public GDK edition folder, the test program is also linked with
//! the GDK's own `xgameruntime.lib` (cfg `decant_with_gdk`), which enables its `online` and
//! `preload` modes, as `DECANT_WITH_GDK` did in the C test. The library goes to the test program
//! only, never to the DLL.

use std::path::PathBuf;

/// The six ordinal exports of the PC runtime, as in the former `xgameruntime.def`.
const ORDINALS: [(&str, u32); 6] =
    [("InitializeApiImpl", 2), ("InitializeApiImplEx", 3), ("InitializeApiImplEx2", 4), ("QueryApiImpl", 5), ("UninitializeApiImpl", 6), ("XErrorReport", 7)];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DECANT_GDK_ROOT");
    println!("cargo:rustc-check-cfg=cfg(decant_with_gdk)");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        // The GNU linker reads ordinals and PRIVATE from a module definition file, which it merges with
        // the one rustc passes for the names. The online test needs the MSVC import library: not here.
        let mut def = String::from("EXPORTS\n");
        for (name, ordinal) in ORDINALS {
            def.push_str(&format!("  {name} @{ordinal}\n"));
        }
        def.push_str("  DllCanUnloadNow PRIVATE\n");
        let path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("ordinals.def");
        std::fs::write(&path, def).expect("write ordinals.def");
        println!("cargo:rustc-cdylib-link-arg={}", path.display());
        return;
    }
    for (name, ordinal) in ORDINALS {
        println!("cargo:rustc-cdylib-link-arg=/EXPORT:{name},@{ordinal}");
    }
    println!("cargo:rustc-cdylib-link-arg=/EXPORT:DllCanUnloadNow,PRIVATE");
    if let Some(root) = std::env::var_os("DECANT_GDK_ROOT").filter(|r| !r.is_empty()) {
        let lib = PathBuf::from(root).join("windows").join("lib").join("x64").join("xgameruntime.lib");
        if lib.is_file() {
            println!("cargo:rustc-cfg=decant_with_gdk");
            println!("cargo:rustc-link-arg-bin=xgameruntime_test={}", lib.display());
        } else {
            println!("cargo:warning=xgameruntime_test: no {}, online mode skipped.", lib.display());
        }
    }
}
