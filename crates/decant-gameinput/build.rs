//! Link settings of `GameInput.dll`.
//!
//! The title imports `GameInput.dll` by ordinal 1, the ordinal of `GameInputCreate` in the system
//! copy. The shim exports the same three names with that ordinal, so a title bound to the system DLL binds to this one.
//! The two COM entries are private exports with ordinals the linker picks.

use std::path::PathBuf;

const ORDINALS: [(&str, Option<u32>); 3] = [("GameInputCreate", Some(1)), ("DllCanUnloadNow", None), ("DllGetClassObject", None)];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        let mut def = String::from("EXPORTS\n");
        for (name, ordinal) in ORDINALS {
            match ordinal {
                Some(n) => def.push_str(&format!("  {name} @{n}\n")),
                None => def.push_str(&format!("  {name} PRIVATE\n")),
            }
        }
        let path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("ordinals.def");
        std::fs::write(&path, def).expect("write ordinals.def");
        println!("cargo:rustc-cdylib-link-arg={}", path.display());
        return;
    }
    for (name, ordinal) in ORDINALS {
        match ordinal {
            Some(n) => println!("cargo:rustc-cdylib-link-arg=/EXPORT:{name},@{n}"),
            None => println!("cargo:rustc-cdylib-link-arg=/EXPORT:{name},PRIVATE"),
        }
    }
}
