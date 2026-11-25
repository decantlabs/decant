//! Exports `DStorageGetFactory` at ordinal 1, as the former `xcrdapi.def` did.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo::rustc-cdylib-link-arg=/EXPORT:DStorageGetFactory,@1");
    } else if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // The GNU linker reads the ordinal from a module definition file and merges it with the one rustc passes.
        let def = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("ordinals.def");
        std::fs::write(&def, "EXPORTS\n  DStorageGetFactory @1\n").expect("write ordinals.def");
        println!("cargo::rustc-cdylib-link-arg={}", def.display());
    }
}
