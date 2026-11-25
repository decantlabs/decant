//! The review rules of the repository: which tracked paths may exist, and what their text may not
//! contain. They were `scripts/check_repository.py`, and keep its allowlists, its order of
//! checks and its messages.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

/// Extensions never tracked: title data, build output, logs.
const FORBIDDEN: &[&str] = &["bk2", "dxbc", "fssm", "spv", "pack", "jsonl", "ll", "exe", "dll", "obj", "lib", "exp", "pdb", "bin", "dmp", "asm", "ini", "log"];
/// Folder or file names that hold private working data.
const PRIVATE: &[&str] = &["out", "index", "runs", "test", "rapport", "codex work"];
const NAMES: &[&str] = &["tri.md", "audit_publication.md", "rapport.md", "settings.json"];
/// `.hlsl`: hand written reference shaders. `.patch`, `.py`: reviewed patches to third party
/// projects and their own scripts (`third_party/` only).
const TEXT: &[&str] = &[".md", ".py", ".ps1", ".cmd", ".yml", ".html", ".hlsl", ".patch", ".rs", ".toml", ".lock"];
const SPECIAL: &[&str] = &["LICENSE", ".github/CODEOWNERS", ".gitattributes", ".gitignore", "third_party/manifest.json", "src/gdk/Makefile", "sandbox/Dockerfile", "tests/detile_vectors.txt",
    // DirectStorage console ABI specs: interface IDs and vtable slots as text, no title content.
    "crates/decant-dstorage/abi/console_hypothesis.abi", "crates/decant-dstorage/abi/pc_hypothesis.abi",
    "crates/decant-dstorage/abi/other_title.abi", "crates/decant-dstorage/abi/retail_title.abi",
    // GFX10 opcode and semantics tables for the direct translator, tab separated, from Mesa and LLVM.
    "docs/gfx10/opcodes.tsv", "docs/gfx10/semantics-arith.tsv",
    // The decoder's round trip vectors, taken from LLVM's GFX10 disassembler tests (Apache 2 with LLVM exception).
    "crates/decant-shader/tests/data/llvm_gfx10.txt"];
/// Rust sources and manifests of the Cargo workspace: reviewed only inside it, which is `crates/` and
/// the files of the workspace root.
const RUST_DIR: &str = "crates/";
const RUST_ROOT_FILES: &[&str] = &["Cargo.toml", "Cargo.lock", "clippy.toml", "rust-toolchain.toml", ".cargo/config.toml"];
const REQUIRED: &[&str] = &[
    "README.md", "LICENSE", "docs/SHADER_TRANSLATOR.md", "build.cmd", ".github/CODEOWNERS", ".github/labels.yml", "SECURITY.md",
];

/// Calls and probes that no Rust source outside `docs/` may name. A D3D12 heap opened on a file
/// mapping keeps the section committed until Windows restarts (measured), a heap opened on an
/// address blocks `VirtualProtect`, decommit and release on its range, and the probe that would try
/// tile mappings on one was never to be run. The names are split so that this file does not contain them.
const FORBIDDEN_CALLS: &[&str] = &[concat!("OpenExistingHeap", "FromFileMapping"), concat!("OpenExistingHeap", "FromAddress"), concat!("tiles", "addr")];

/// Local user paths, tokens, private keys and inline secrets.
static PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)[a-z]:[\\/](?:users[\\/]|jeux[\\/]|steamlibrary[\\/])",
        r"(?i)gh[pousr]_[a-z0-9]{24,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"AKIA[A-Z0-9]{16}",
        r"(?i)github_pat_[a-z0-9_]{30,}",
        r#"(?im)^\s*(?:password|api_key|access_token)\s*=\s*["'][^"']{8,}"#,
    ]
    .iter()
    .map(|p| Regex::new(p).unwrap())
    .collect()
});

/// `PurePath.suffix`: the last `.xxx` of the final name, empty for a name that starts with its
/// only dot or ends with a dot.
pub fn suffix(name: &str) -> &str {
    let last = name.rsplit(['/', '\\']).next().unwrap_or(name);
    match last.rfind('.') {
        Some(i) if i > 0 && i < last.len() - 1 => &last[i..],
        _ => "",
    }
}

fn file_name(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

/// What the check finds.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every path passed: the number of paths reviewed.
    Reviewed(usize),
    /// The paths that need review, in the order found, then the missing required files.
    Review(Vec<String>),
}

/// A stop the script took with a traceback: a file that is not UTF 8.
#[derive(Debug, PartialEq, Eq)]
pub struct Stop(pub String);

/// `read_text(encoding="utf-8-sig")`: UTF 8 without its byte order mark, line ends made `\n`.
fn read_text(path: &Path, name: &str) -> Result<String, Stop> {
    let bytes = std::fs::read(path).map_err(|e| Stop(format!("{name}: {e}")))?;
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
    let text = String::from_utf8(bytes.to_vec()).map_err(|e| Stop(format!("UnicodeDecodeError: {name}: {e}")))?;
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// Reviews the tracked paths `names` of the repository at `root`.
pub fn check(root: &Path, names: &[String]) -> Result<Outcome, Stop> {
    let mut errors = Vec::new();
    for name in names {
        let path = root.join(name);
        let lower = name.to_lowercase();
        let parts: HashSet<&str> = lower.split(['/', '\\']).filter(|p| !p.is_empty()).collect();
        let link = std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink());
        let meta = std::fs::metadata(&path).ok().filter(|m| m.is_file());
        let ext = suffix(name).to_lowercase();
        if !TEXT.contains(&ext.as_str()) && !SPECIAL.contains(&name.as_str()) {
            errors.push(name.clone());
            continue;
        }
        // Every tool of the repository is Rust: Python and patches are accepted in third party sources only.
        if [".patch", ".py"].contains(&ext.as_str()) && !name.starts_with("third_party/") {
            errors.push(name.clone());
            continue;
        }
        if [".rs", ".toml", ".lock"].contains(&ext.as_str()) && !(name.starts_with(RUST_DIR) || RUST_ROOT_FILES.contains(&name.as_str())) {
            errors.push(name.clone());
            continue;
        }
        let forbidden = ext.strip_prefix('.').is_some_and(|e| FORBIDDEN.contains(&e));
        if link || meta.is_none() || forbidden || PRIVATE.iter().any(|p| parts.contains(p)) || NAMES.contains(&file_name(&lower)) {
            errors.push(name.clone());
            continue;
        }
        let content = read_text(&path, name)?;
        if content.contains('\0') || PATTERNS.iter().any(|p| p.is_match(&content)) || (ext == ".rs" && !name.starts_with("docs/") && FORBIDDEN_CALLS.iter().any(|c| content.contains(c))) {
            errors.push(name.clone());
        }
    }
    let present: HashSet<&str> = names.iter().map(String::as_str).collect();
    let missing: BTreeSet<&str> = REQUIRED.iter().copied().filter(|r| !present.contains(r)).collect();
    errors.extend(missing.into_iter().map(str::to_owned));
    Ok(if errors.is_empty() { Outcome::Reviewed(names.len()) } else { Outcome::Review(errors) })
}

/// `git ls-files -z` in `root`, decoded as UTF 8.
pub fn tracked(root: &Path) -> Result<Vec<String>, String> {
    let out = std::process::Command::new("git").arg("ls-files").arg("-z").current_dir(root).output().map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!("CalledProcessError: git ls-files -z returned {}", out.status));
    }
    out.stdout.split(|&b| b == 0).filter(|n| !n.is_empty()).map(|n| String::from_utf8(n.to_vec()).map_err(|e| format!("UnicodeDecodeError: {e}"))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tree(std::path::PathBuf);

    impl Tree {
        fn new(tag: &str) -> Tree {
            let dir = std::env::temp_dir().join(format!("repo-check-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Tree(dir)
        }

        fn file(&self, name: &str, text: &[u8]) -> String {
            let path = self.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
            name.to_owned()
        }

        fn required(&self) -> Vec<String> {
            REQUIRED.iter().map(|r| self.file(r, b"text\n")).collect()
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn suffixes_follow_pathlib() {
        assert_eq!(suffix("a/b.tar.gz"), ".gz");
        assert_eq!(suffix(".gitignore"), "");
        assert_eq!(suffix("dir.d/name"), "");
        assert_eq!(suffix("name."), "");
        assert_eq!(suffix("x/.cargo/config.toml"), ".toml");
    }

    #[test]
    fn a_clean_tree_is_reviewed() {
        let t = Tree::new("clean");
        let mut names = t.required();
        names.push(t.file("docs/notes.md", b"\xEF\xBB\xBFsome text\r\nmore text\r\n"));
        names.push(t.file("third_party/x/helper.py", b"print('third party')\n"));
        names.push(t.file("crates/x/src/lib.rs", b"fn main() {}\n"));
        names.push(t.file("third_party/x/fix.patch", b"diff\n"));
        assert_eq!(check(&t.0, &names), Ok(Outcome::Reviewed(names.len())));
    }

    #[test]
    fn each_rule_reports_its_path_in_order() {
        let t = Tree::new("rules");
        let mut names = t.required();
        names.retain(|n| n != "docs/SHADER_TRANSLATOR.md");
        let bad = [
            t.file("data/shaders.spv.md.exe", b"x"),
            t.file("src/fix.patch", b"x"),
            t.file("src/other.c", b"x"),
            t.file("tools/build.rs", b"x"),
            t.file("out/notes.md", b"x"),
            t.file("docs/RAPPORT.md", b"x"),
            // Split, so that this source file does not match the pattern it tests.
            t.file("docs/key.md", concat!("token gh", "p_abcdefghijklmnopqrstuvwxyz0123\n").as_bytes()),
            t.file("docs/path.md", b"see C:\\Users\\someone\\file\n"),
            t.file("docs/secret.md", b"  password = 'abcdefgh1'\n"),
            t.file("docs/nul.md", b"a\0b"),
        ];
        names.extend(bad.iter().cloned());
        names.push(t.file("docs/fine.md", b"password = 'short'\n"));
        names.push("docs/gone.md".to_owned());
        let mut expected: Vec<String> = bad.to_vec();
        expected.push("docs/gone.md".to_owned());
        expected.push("docs/SHADER_TRANSLATOR.md".to_owned());
        assert_eq!(check(&t.0, &names), Ok(Outcome::Review(expected)));
    }

    #[test]
    fn rust_sources_may_not_name_the_heaps_that_leak_or_pin_memory() {
        let t = Tree::new("heaps");
        let mut names = t.required();
        names.push(t.file("crates/x/src/fine.rs", b"fn main() { /* CreateHeap */ }\n"));
        // Text under docs/ may name them. The names are split so that this source does not contain them.
        names.push(t.file("docs/spec.md", concat!("// OpenExistingHeap", "FromFileMapping\n").as_bytes()));
        assert_eq!(check(&t.0, &names), Ok(Outcome::Reviewed(names.len())));
        let bad = [
            t.file("crates/x/src/a.rs", concat!("fn f() { d.OpenExistingHeap", "FromFileMapping(h); }\n").as_bytes()),
            t.file("crates/x/src/b.rs", concat!("fn f() { d.OpenExistingHeap", "FromAddress1(p, n); }\n").as_bytes()),
            t.file("crates/x/src/c.rs", concat!("// the tiles", "addr probe\n").as_bytes()),
        ];
        names.extend(bad.iter().cloned());
        assert_eq!(check(&t.0, &names), Ok(Outcome::Review(bad.to_vec())));
    }

    #[test]
    fn python_is_only_accepted_in_third_party() {
        let t = Tree::new("python");
        let mut names = t.required();
        names.push(t.file("third_party/x/helper.py", b"print('ok')\n"));
        assert_eq!(check(&t.0, &names), Ok(Outcome::Reviewed(names.len())));
        let own = [t.file("src/tool.py", b"print('ok')\n"), t.file("src/shaders/tools/other.PY", b"print('ok')\n")];
        names.extend(own.iter().cloned());
        assert_eq!(check(&t.0, &names), Ok(Outcome::Review(own.to_vec())));
    }

    #[test]
    fn a_file_that_is_not_utf8_stops_the_check() {
        let t = Tree::new("latin");
        let mut names = t.required();
        names.push(t.file("latin.md", b"caf\xe9\n"));
        assert!(check(&t.0, &names).unwrap_err().0.starts_with("UnicodeDecodeError"));
    }

    #[test]
    fn binary_files_are_sent_to_review() {
        let t = Tree::new("binary");
        let mut names = t.required();
        names.push(t.file("docs/images/picture.jpg", &[0; 64]));
        assert_eq!(check(&t.0, &names), Ok(Outcome::Review(vec!["docs/images/picture.jpg".to_owned()])));
    }
}
