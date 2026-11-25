//! Build script of `dstorage_xs.dll` and `xcrdapi.dll`.
//!
//! The DLLs export `DStorageGetFactory` at ordinal 1, as the module definition files of the former
//! `build_storage.cmd` did (`dstorage_xs.def`, `xcrdapi.def`).
//!
//! GDeflate decoding needs libdeflate with the NVIDIA GDeflate additions, which the C++ build
//! compiled from the user's checkout of microsoft/DirectStorage. Here the same checkout is named
//! by `DECANT_GDEFLATE_ROOT` (its `GDeflate` folder, the one holding `3rdparty/libdeflate`): when it
//! is set and complete, its C sources are compiled into a static library and linked, and the
//! `decant_gdeflate` cfg is set. Without it the crate builds all the same and GDeflate requests on
//! the CPU path answer `E_NOTIMPL`. The PC DirectStorage runtime is never needed to build: it is
//! loaded at run time.

use std::path::PathBuf;

/// Compiled for the decoders of the DLL.
const DECODE: [&str; 5] = ["adler32", "deflate_decompress", "gdeflate_decompress", "utils", "zlib_decompress"];
/// Compiled for the test programs, which make their own compressed data.
const COMPRESS: [&str; 3] = ["deflate_compress", "gdeflate_compress", "zlib_compress"];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=DECANT_GDEFLATE_ROOT");
    println!("cargo::rustc-check-cfg=cfg(decant_gdeflate)");
    let target = |key: &str| std::env::var(key).unwrap_or_default();
    if target("CARGO_CFG_TARGET_OS") != "windows" {
        return;
    }
    // MSVC takes the ordinal as an option. The GNU linker reads a module definition file: rustc already
    // passes one with the names, this one adds the ordinal and the linker merges the two.
    if target("CARGO_CFG_TARGET_ENV") == "msvc" {
        println!("cargo::rustc-cdylib-link-arg=/EXPORT:DStorageGetFactory,@1");
    } else {
        let def = PathBuf::from(target("OUT_DIR")).join("ordinals.def");
        std::fs::write(&def, "EXPORTS\n  DStorageGetFactory @1\n").expect("write ordinals.def");
        println!("cargo::rustc-cdylib-link-arg={}", def.display());
    }
    if target("CARGO_CFG_TARGET_ARCH") == "x86_64" {
        gdeflate();
    }
}

fn gdeflate() {
    let Some(root) = std::env::var_os("DECANT_GDEFLATE_ROOT").filter(|r| !r.is_empty()) else {
        return;
    };
    let dir = PathBuf::from(root).join("3rdparty").join("libdeflate");
    if !dir.join("lib").join("gdeflate_decompress.c").is_file() {
        println!("cargo::warning=DECANT_GDEFLATE_ROOT has no 3rdparty/libdeflate/lib/gdeflate_decompress.c: GDeflate stays off");
        return;
    }
    let mut build = cc::Build::new();
    build.include(&dir).opt_level(2).debug(false).warnings(false).flag_if_supported("/std:c11");
    for name in DECODE.iter().chain(&COMPRESS) {
        let file = dir.join("lib").join(format!("{name}.c"));
        println!("cargo::rerun-if-changed={}", file.display());
        build.file(file);
    }
    build.file(dir.join("lib").join("x86").join("cpu_features.c"));
    build.compile("decant_storage_decode");
    println!("cargo::rustc-cfg=decant_gdeflate");
}
