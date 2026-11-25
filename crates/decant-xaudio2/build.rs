//! Gives `decant_xaudio2.dll` the export ordinals of the PC `XAudio2_9.dll`, as the former
//! `decant_xa2.def` did. Console modules import by ordinal (`Bink2Scarlett.dll` in the package imports
//! `XAudio2_9.dll` #1, which is `XAudio2Create` on PC). The functions in `src/exports.rs` carry the
//! names. This only adds the ordinal to each of them. Other targets: nothing to do.

/// Export name and ordinal, the same table as the former `decant_xa2.def`.
const ORDINALS: &[(&str, u32)] = &[
    ("XAudio2Create", 1),
    ("CreateAudioReverb", 2),
    ("CreateAudioVolumeMeter", 3),
    ("XAudio2CreateWithVersionInfo", 9),
    ("DecantXa2GetStats", 20),
    ("DecantXa2ResetStats", 21),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if !windows {
        return;
    }
    if msvc {
        for (name, ordinal) in ORDINALS {
            println!("cargo:rustc-cdylib-link-arg=/EXPORT:{name},@{ordinal}");
        }
    } else {
        // The GNU linker reads the ordinals from a module definition file, which it merges with the one
        // rustc passes for the names.
        let mut def = String::from("EXPORTS\n");
        for (name, ordinal) in ORDINALS {
            def.push_str(&format!("  {name} @{ordinal}\n"));
        }
        let path = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("ordinals.def");
        std::fs::write(&path, def).expect("write ordinals.def");
        println!("cargo:rustc-cdylib-link-arg={}", path.display());
    }
}
