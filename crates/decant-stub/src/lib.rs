//! Stand in DLLs for console modules that a title imports and nothing provides.
//!
//! The Windows loader stops at the first import it cannot resolve and does not say which one. A
//! stand in DLL per missing module, with one export per function the title imports, lets the title
//! load. Each export writes its name to the debugger the first few times it is called and returns
//! `E_NOTIMPL`, so a run shows which functions the title really calls. The functions are listed by
//! `audit_xbox_imports` (the `--out` folder holds `rapport_imports.json`).
//!
//! This writes one small Rust `cdylib` project per module: `Cargo.toml`, `build.rs`,
//! `.cargo/config.toml` and `src/lib.rs`. A project has no dependency and its own empty
//! `[workspace]`, so it builds on its own with cargo, outside this workspace. [`build`] runs that
//! build and returns the DLL. No generated project and no DLL goes into the repository: the export
//! lists come from the title, only this generator is committed.
//!
//! One return convention fits every signature on x64 (arguments in registers, caller cleans the
//! stack), so every stub is `HRESULT f(void)`. A function that returns a pointer or a count gets
//! `E_NOTIMPL` in the return register. That is wrong for it and is what the log is for.
//!
//! Ordinals: a console module imports some functions by number. Such an export is a stub named
//! `decant_stub_<i>` and the generated `build.rs` pins its number with `/EXPORT:decant_stub_<i>,@<n>`.
//! The C runtime is linked statically (`+crt-static`), so the DLL needs nothing installed.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// How many calls of one export are logged, so a hot function does not flood the debugger.
pub const LOG_LIMIT: u32 = 8;

/// One export of a stand in module.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Export {
    Name(String),
    Ordinal(u16),
}

/// A module to stand in for.
#[derive(Debug, PartialEq, Eq)]
pub struct Module {
    /// File name as imported, for example `xmem.dll`.
    pub file: String,
    pub exports: Vec<Export>,
}

impl Module {
    /// The file name without its extension. It names the generated project and the library.
    pub fn stem(&self) -> &str {
        self.file.rsplit_once('.').map_or(self.file.as_str(), |(stem, _)| stem)
    }
}

/// A DLL file name that is also a valid cargo library name, so the built file has the exact name.
fn safe_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let Some(stem) = name.get(..name.len().saturating_sub(4)) else { return false };
    lower.ends_with(".dll")
        && !stem.is_empty()
        && stem.len() <= 60
        && !lower.starts_with("api_ms_")
        && !lower.starts_with("ext_ms_")
        && !stem.starts_with(|c: char| c.is_ascii_digit())
        && stem.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Characters of C and C++ decorated export names, and nothing that could leave a Rust string.
fn safe_export(name: &str) -> bool {
    !name.is_empty() && name.len() <= 512 && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '?' | '@' | '$' | '.'))
}

/// The modules the report lists as absent (neither in the package, on this PC nor in `--shims`),
/// each with every function it imports. `only` keeps the named modules, case insensitively
/// (all of them when empty). Errors name each unusable module or function instead of skipping it.
pub fn modules(report: &str, only: &[String]) -> Result<Vec<Module>, String> {
    let root: Value = serde_json::from_str(report).map_err(|e| format!("the report is not JSON: {e}"))?;
    let dlls = root.get("dll").and_then(Value::as_object).ok_or("the report has no \"dll\" table")?;
    let wanted: Vec<String> = only.iter().map(|n| n.to_ascii_lowercase()).collect();
    let mut out = Vec::new();
    let mut problems = Vec::new();
    for (key, entry) in dlls {
        if entry.get("origine").and_then(Value::as_str) != Some("absente") {
            continue;
        }
        let file = entry.get("nom").and_then(Value::as_str).unwrap_or(key);
        if !wanted.is_empty() && !wanted.iter().any(|w| w == &file.to_ascii_lowercase()) {
            continue;
        }
        if !safe_file(file) {
            problems.push(format!("module name not usable: {file:?}"));
            continue;
        }
        let mut exports = BTreeMap::new();
        for label in entry.get("fonctions").and_then(Value::as_object).into_iter().flat_map(|f| f.keys()) {
            let export = match label.strip_prefix('#') {
                Some(ordinal) => match ordinal.parse::<u16>() {
                    Ok(o) if o != 0 => Export::Ordinal(o),
                    _ => {
                        problems.push(format!("{file}: bad ordinal {label:?}"));
                        continue;
                    }
                },
                None if safe_export(label) => Export::Name(label.clone()),
                None => {
                    problems.push(format!("{file}: export name not usable: {label:?}"));
                    continue;
                }
            };
            exports.insert(export.clone(), export);
        }
        if !exports.is_empty() {
            out.push(Module { file: file.to_owned(), exports: exports.into_values().collect() });
        }
    }
    if !problems.is_empty() {
        return Err(problems.join("\n"));
    }
    out.sort_by_key(|m| m.file.to_ascii_lowercase());
    Ok(out)
}

fn export_label(e: &Export) -> String {
    match e {
        Export::Name(n) => n.clone(),
        Export::Ordinal(o) => format!("#{o}"),
    }
}

/// The `Cargo.toml` of one stand in project: package named after the module, a `cdylib` whose
/// library name gives the exact DLL file name, no dependency, and its own empty workspace.
pub fn cargo_toml(m: &Module) -> String {
    let stem = m.stem();
    format!(
        "# Generated by decant_stubgen. Stand in for {file}.\n[package]\nname = \"{stem}\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\nbuild = \"build.rs\"\n\n[lib]\nname = \"{stem}\"\ncrate-type = [\"cdylib\"]\n\n[profile.release]\npanic = \"abort\"\n\n# An empty workspace of its own, so the project builds outside any other workspace.\n[workspace]\n",
        file = m.file
    )
}

/// The `.cargo/config.toml`: static C runtime. The `--build` flag also sets it through the
/// environment, which wins over a `RUSTFLAGS` of the caller.
pub fn cargo_config() -> String {
    "# Generated by decant_stubgen.\n[target.'cfg(all(windows, target_env = \"msvc\"))']\nrustflags = [\"-C\", \"target-feature=+crt-static\"]\n".to_owned()
}

/// The `build.rs`: pins the ordinal of every export the title imports by number. Other targets
/// have nothing to do.
pub fn build_rs(m: &Module) -> String {
    let mut s = String::from("//! Generated by decant_stubgen. Pins the export ordinals.\n\nfn main() {\n    println!(\"cargo:rerun-if-changed=build.rs\");\n    let msvc = std::env::var(\"CARGO_CFG_TARGET_ENV\").as_deref() == Ok(\"msvc\");\n    if !msvc {\n        return;\n    }\n");
    for (i, e) in m.exports.iter().enumerate() {
        if let Export::Ordinal(o) = e {
            let _ = writeln!(s, "    println!(\"cargo:rustc-cdylib-link-arg=/EXPORT:decant_stub_{i},@{o}\");");
        }
    }
    s.push_str("}\n");
    s
}

/// The Rust source of one stand in module.
pub fn lib_rs(m: &Module) -> String {
    let n = m.exports.len();
    let mut s = String::new();
    let _ = writeln!(s, "//! Generated by decant_stubgen. Stand in for {}: {n} export(s), each logs its name and returns E_NOTIMPL.\n", m.file);
    s.push_str("#![allow(non_snake_case)]\n\nuse std::sync::atomic::{AtomicU32, Ordering};\n\n");
    let _ = writeln!(s, "const MODULE: &str = \"{}\";\nconst LOG_LIMIT: u32 = {LOG_LIMIT};", m.file);
    s.push_str("const E_NOTIMPL: i32 = 0x8000_4001_u32 as i32;\nconst ERROR_CALL_NOT_IMPLEMENTED: u32 = 120;\n\n");
    s.push_str("#[cfg(windows)]\n#[link(name = \"kernel32\")]\nunsafe extern \"system\" {\n    fn OutputDebugStringA(text: *const u8);\n    fn SetLastError(code: u32);\n}\n\n");
    s.push_str("#[cfg(not(windows))]\nunsafe fn OutputDebugStringA(_text: *const u8) {}\n#[cfg(not(windows))]\nunsafe fn SetLastError(_code: u32) {}\n\n");
    let _ = writeln!(s, "static NAMES: [&str; {n}] = [");
    for e in &m.exports {
        let _ = writeln!(s, "    \"{}\",", export_label(e));
    }
    s.push_str("];\n");
    let _ = writeln!(s, "static CALLS: [AtomicU32; {n}] = [const {{ AtomicU32::new(0) }}; {n}];\n");
    s.push_str("fn hit(i: usize) -> i32 {\n    let n = CALLS[i].fetch_add(1, Ordering::Relaxed).saturating_add(1);\n    if n <= LOG_LIMIT {\n        let line = format!(\"[decant_stub] {MODULE}!{} (call {n})\\n\\0\", NAMES[i]);\n        // SAFETY: `line` is NUL terminated and lives through the call.\n        unsafe { OutputDebugStringA(line.as_ptr()) };\n    }\n    // SAFETY: plain Win32 call without pointers.\n    unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED) };\n    E_NOTIMPL\n}\n\n");
    for (i, e) in m.exports.iter().enumerate() {
        match e {
            Export::Name(name) => {
                let _ = writeln!(s, "#[unsafe(export_name = \"{name}\")]\npub extern \"system\" fn decant_stub_{i}() -> i32 {{\n    hit({i})\n}}\n");
            }
            Export::Ordinal(_) => {
                let _ = writeln!(s, "#[unsafe(no_mangle)]\npub extern \"system\" fn decant_stub_{i}() -> i32 {{\n    hit({i})\n}}\n");
            }
        }
    }
    s
}

/// Every generated file as (path relative to the output folder, text): the four files of each
/// project, under a folder named after the module.
pub fn files(modules: &[Module]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for m in modules {
        let stem = m.stem();
        out.push((format!("{stem}/Cargo.toml"), cargo_toml(m)));
        out.push((format!("{stem}/build.rs"), build_rs(m)));
        out.push((format!("{stem}/.cargo/config.toml"), cargo_config()));
        out.push((format!("{stem}/src/lib.rs"), lib_rs(m)));
    }
    out
}

/// Writes every project below `out`. Returns the written paths.
pub fn write(modules: &[Module], out: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    for (name, content) in files(modules) {
        let path = out.join(&name);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(&path, content).map_err(|e| format!("{name}: {e}"))?;
        paths.push(path);
    }
    Ok(paths)
}

/// Builds the project of `m` (written below `out`) with `cargo build --release` and copies the DLL
/// to `out`, next to the projects. Returns the DLL path.
pub fn build(m: &Module, out: &Path) -> Result<PathBuf, String> {
    let project = out.join(m.stem());
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .args(["build", "--release", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env("RUSTFLAGS", "-C target-feature=+crt-static")
        .status()
        .map_err(|e| format!("cargo could not start: {e}"))?;
    if !status.success() {
        return Err(format!("{}: cargo build failed ({status})", m.file));
    }
    let built = project.join("target").join("release").join(&m.file);
    let dest = out.join(&m.file);
    std::fs::copy(&built, &dest).map_err(|e| format!("{}: {e}", built.display()))?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPORT: &str = r##"{
      "dossier": "x", "fichiers": {}, "illisibles": {},
      "dll": {
        "zork_xs.dll": {"nom": "Zork_xs.dll", "origine": "absente", "differe": false, "importee_par": ["a.dll"],
                      "fonctions": {"ZorkFree": false, "ZorkAlloc": false, "#7": false}},
        "kernel32.dll": {"nom": "KERNEL32.dll", "origine": "PC", "differe": false, "importee_par": ["a.dll"], "fonctions": {"Sleep": true}},
        "quux.dll": {"nom": "quux.dll", "origine": "absente", "differe": false, "importee_par": ["a.dll"],
                      "fonctions": {"?Build@QuuxBuilder@@QEAAXXZ": false}}
      }
    }"##;

    fn zork() -> Module {
        modules(REPORT, &["zork_xs.dll".into()]).unwrap().remove(0)
    }

    #[test]
    fn only_absent_modules_get_a_stand_in_and_exports_are_sorted() {
        let mods = modules(REPORT, &[]).unwrap();
        assert_eq!(mods.iter().map(|m| m.file.as_str()).collect::<Vec<_>>(), ["quux.dll", "Zork_xs.dll"]);
        assert_eq!(mods[1].exports, [Export::Name("ZorkAlloc".into()), Export::Name("ZorkFree".into()), Export::Ordinal(7)]);
        assert_eq!(mods[1].stem(), "Zork_xs");
    }

    #[test]
    fn only_filter_is_case_insensitive() {
        let mods = modules(REPORT, &["ZORK_XS.DLL".into()]).unwrap();
        assert_eq!(mods.len(), 1);
        assert!(modules(REPORT, &["nothere.dll".into()]).unwrap().is_empty());
    }

    #[test]
    fn cargo_toml_names_the_package_and_the_dll_and_has_its_own_workspace() {
        let t = cargo_toml(&zork());
        assert!(t.contains("name = \"Zork_xs\""));
        assert!(t.contains("[lib]\nname = \"Zork_xs\"\ncrate-type = [\"cdylib\"]"));
        assert!(t.contains("\n[workspace]\n"));
        assert!(!t.contains("[dependencies]") && !t.contains("path ="));
        assert!(cargo_config().contains("+crt-static"));
    }

    #[test]
    fn build_rs_pins_only_the_ordinal_exports() {
        let b = build_rs(&zork());
        assert_eq!(b.matches("/EXPORT:").count(), 1);
        assert!(b.contains("/EXPORT:decant_stub_2,@7"));
    }

    #[test]
    fn lib_rs_has_one_stub_and_one_name_per_export() {
        let c = lib_rs(&zork());
        assert!(c.contains("static NAMES: [&str; 3]"));
        assert!(c.contains("\"ZorkAlloc\",") && c.contains("\"#7\","));
        assert_eq!(c.matches("pub extern \"system\" fn decant_stub_").count(), 3);
        assert!(c.contains("#[unsafe(export_name = \"ZorkFree\")]"));
        assert!(c.contains("#[unsafe(no_mangle)]\npub extern \"system\" fn decant_stub_2()"));
        assert!(c.contains("const MODULE: &str = \"Zork_xs.dll\";") && c.contains("const LOG_LIMIT: u32 = 8;"));
        assert!(c.contains("E_NOTIMPL") && c.contains("OutputDebugStringA"));
        let q = lib_rs(&modules(REPORT, &["quux.dll".into()]).unwrap()[0]);
        assert!(q.contains("export_name = \"?Build@QuuxBuilder@@QEAAXXZ\""));
    }

    #[test]
    fn unsafe_names_are_refused_not_skipped() {
        let bad = r##"{"dll": {"a.dll": {"nom": "a.dll", "origine": "absente", "fonctions": {"ok": false, "evil\"; x": false, "#0": false}},
                                "..\\b.dll": {"nom": "..\\b.dll", "origine": "absente", "fonctions": {"f": false}}}}"##;
        let e = modules(bad, &[]).unwrap_err();
        assert!(e.contains("export name not usable"), "{e}");
        assert!(e.contains("bad ordinal"), "{e}");
        assert!(e.contains("module name not usable"), "{e}");
    }

    #[test]
    fn api_sets_and_non_library_names_are_not_stood_in_for() {
        assert!(!safe_file("api_ms_win_core_file_l1_1_0.dll"));
        assert!(!safe_file("readme.txt"));
        assert!(!safe_file(".dll"));
        assert!(!safe_file("a-b.dll"));
        assert!(!safe_file("1abc.dll"));
        assert!(safe_file("d3d12_xs.dll"));
    }

    #[test]
    fn files_are_four_per_module_and_no_c_is_written() {
        let mods = modules(REPORT, &[]).unwrap();
        let all = files(&mods);
        assert_eq!(all.len(), 4 * mods.len());
        let names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Zork_xs/src/lib.rs") && names.contains(&"quux/.cargo/config.toml"));
        assert!(names.iter().all(|n| !n.ends_with(".c") && !n.ends_with(".def") && !n.ends_with(".cmd")));
    }
}
