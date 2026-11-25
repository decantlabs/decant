//! The third party pieces kept in `third_party/` (see `third_party/README.md`): list them, check
//! the manifest against the folders, and put the patches in place. This was
//! `scripts/third_party.py`, with the same checks and messages.
//!
//! `setup` never downloads anything. Each checkout is a git clone the user made: it is checked to
//! be at the commit named in `third_party/manifest.json`, then the patch is applied. A patch that
//! is already applied is left alone.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Map, Value};

/// The `items` of `third_party/manifest.json`, in file order.
pub fn load_manifest(third: &Path) -> Result<Map<String, Value>, String> {
    let path = third.join("manifest.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let value: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    match value.get("items") {
        Some(Value::Object(items)) => Ok(items.clone()),
        Some(_) => Err("AttributeError: items is not an object".to_owned()),
        None => Err("KeyError: 'items'".to_owned()),
    }
}

/// Python's truth value of a JSON value.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `str(value)` for the message of a missing licence.
fn text_of(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "None".to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => if *b { "True" } else { "False" }.to_owned(),
        Some(other) => other.to_string(),
    }
}

/// Every file under `folder`, as a path relative to it with `/`, symbolic links to folders not
/// followed (`Path.rglob("*")` then `is_file()`).
fn files_under(folder: &Path) -> Vec<String> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_link = entry.file_type().is_ok_and(|t| t.is_symlink());
            if path.is_file() {
                let rel = path.strip_prefix(base).unwrap_or(&path);
                out.push(rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/"));
            } else if path.is_dir() && !is_link {
                walk(base, &path, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(folder, folder, &mut out);
    out
}

/// A text file read as UTF 8 in text mode.
fn read_text(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// The problems of `third`: every folder listed, every listed file present and nothing more,
/// every licence named in its README, every patch listed.
pub fn check(third: &Path) -> Result<Vec<String>, String> {
    let mut problems = Vec::new();
    let items = load_manifest(third)?;
    let mut dirs: Vec<String> = std::fs::read_dir(third)
        .map_err(|e| format!("{}: {e}", third.display()))?
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    // pathlib sorts paths case insensitively on Windows.
    dirs.sort_by_key(|d| if cfg!(windows) { d.to_lowercase() } else { d.clone() });
    for dir in dirs {
        if !items.contains_key(&dir) {
            problems.push(format!("{dir}: not in manifest.json"));
        }
    }
    for (name, item) in &items {
        let folder = third.join(name);
        for key in ["licence", "licence_file", "upstream", "commit", "files"] {
            if !truthy(item.get(key)) {
                problems.push(format!("{name}: missing '{key}'"));
            }
        }
        let commit_len = match item.get("commit") {
            None => 0,
            Some(Value::String(s)) => s.chars().count(),
            Some(Value::Array(a)) => a.len(),
            Some(Value::Object(o)) => o.len(),
            Some(other) => return Err(format!("TypeError: object of type {other} has no len()")),
        };
        if commit_len != 40 {
            problems.push(format!("{name}: commit must be a full 40-character hash"));
        }
        let listed: std::collections::BTreeSet<String> =
            item.get("files").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect()).unwrap_or_default();
        let present: std::collections::BTreeSet<String> = if folder.is_dir() { files_under(&folder).into_iter().collect() } else { Default::default() };
        problems.extend(listed.difference(&present).map(|f| format!("{name}: listed but missing: {f}")));
        problems.extend(present.difference(&listed).map(|f| format!("{name}: present but not listed: {f}")));
        let licence = item.get("licence").and_then(Value::as_str).unwrap_or("?");
        let named = folder.join("README.md").is_file() && read_text(&folder.join("README.md")).is_some_and(|t| t.contains(licence));
        if !named {
            problems.push(format!("{name}: README.md must name the licence {}", text_of(item.get("licence"))));
        }
        if truthy(item.get("patch")) && !item.get("patch").and_then(Value::as_str).is_some_and(|p| listed.contains(p)) {
            problems.push(format!("{name}: patch file not listed"));
        }
    }
    Ok(problems)
}

/// The lines of `list`.
pub fn list(items: &Map<String, Value>) -> Vec<String> {
    items
        .iter()
        .map(|(name, item)| {
            let field = |k: &str| item.get(k).map_or(String::new(), |v| text_of(Some(v)));
            let commit: String = field("commit").chars().take(12).collect();
            format!("{name:14} {:9} {} @ {commit}", field("licence"), field("upstream"))
        })
        .collect()
}

/// `str(path)` of a resolved path on Windows: backslashes, no `\\?\` prefix.
pub fn display(path: &Path) -> String {
    let text = path.to_string_lossy().into_owned();
    let text = text.strip_prefix(r"\\?\").map(str::to_owned).unwrap_or(text);
    if cfg!(windows) { text.replace('/', "\\") } else { text }
}

/// `Path(text).resolve()`: absolute, with links followed when the path exists.
pub fn resolve(text: &str) -> PathBuf {
    let path = Path::new(text);
    match std::fs::canonicalize(path) {
        Ok(p) => PathBuf::from(display(&p)),
        Err(_) => std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()),
    }
}

/// `git -C <checkout> <args>`: success, standard output and error stream as text.
fn git(checkout: &Path, args: &[&str]) -> (bool, String, String) {
    match Command::new("git").arg("-C").arg(checkout).args(args).output() {
        Ok(o) => (o.status.success(), decant_shader_core::py::read_text(&o.stdout), decant_shader_core::py::read_text(&o.stderr)),
        Err(e) => (false, String::new(), e.to_string()),
    }
}

fn last_chars(text: &str, n: usize) -> String {
    let chars: Vec<char> = text.trim().chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

/// One checkout: checks its commit, then applies the patch unless it is already applied. Returns
/// the error, if any, and prints progress through `say`.
pub fn setup_one(third: &Path, name: &str, item: &Value, checkout: &str, dry_run: bool, say: &mut dyn FnMut(String)) -> Option<String> {
    let checkout = resolve(checkout);
    let shown = display(&checkout);
    let field = |k: &str| item.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
    let (commit, upstream, licence) = (field("commit"), field("upstream"), field("licence"));
    let (ok, head, _) = git(&checkout, &["rev-parse", "HEAD"]);
    if !ok {
        return Some(format!("{name}: {shown} is not a git checkout"));
    }
    let head = head.trim();
    if head != commit {
        let short = |s: &str| s.chars().take(12).collect::<String>();
        return Some(format!("{name}: {shown} is at {}, expected {} ({upstream})", short(head), short(&commit)));
    }
    let patch = display(&third.join(name).join(field("patch")));
    if git(&checkout, &["apply", "--reverse", "--check", &patch]).0 {
        say(format!("{name}: patch already applied in {shown}"));
        return None;
    }
    let (ok, _, err) = git(&checkout, &["apply", "--check", "--whitespace=nowarn", &patch]);
    if !ok {
        return Some(format!("{name}: the patch does not apply cleanly in {shown}: {}", last_chars(&err, 300)));
    }
    if dry_run {
        say(format!("{name}: patch would apply cleanly in {shown}"));
        return None;
    }
    let (ok, _, err) = git(&checkout, &["apply", "--whitespace=nowarn", &patch]);
    if !ok {
        return Some(format!("{name}: applying the patch failed: {}", last_chars(&err, 300)));
    }
    say(format!("{name}: patch applied in {shown} ({licence})"));
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_third() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party")
    }

    /// Was `ThirdParty.test_manifest_matches_directory` in `tests/test_tools.py`.
    #[test]
    fn manifest_matches_directory() {
        assert_eq!(check(&repo_third()).unwrap(), Vec::<String>::new());
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("repo-check-third-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Was `ThirdParty.test_unlisted_directory_is_reported` in `tests/test_tools.py`.
    #[test]
    fn unlisted_directory_is_reported() {
        let third = scratch("stray");
        std::fs::create_dir_all(third.join("listed")).unwrap();
        std::fs::write(third.join("listed/README.md"), "Licence: MIT-0\n").unwrap();
        std::fs::create_dir_all(third.join("stray")).unwrap();
        let item = serde_json::json!({"licence": "MIT-0", "licence_file": "README.md", "upstream": "https://example.invalid/x", "commit": "0".repeat(40), "files": ["README.md"]});
        std::fs::write(third.join("manifest.json"), serde_json::json!({"items": {"listed": item}}).to_string()).unwrap();
        assert_eq!(check(&third).unwrap(), ["stray: not in manifest.json"]);
        std::fs::remove_dir_all(&third).unwrap();
    }

    #[test]
    fn every_problem_is_named() {
        let third = scratch("problems");
        std::fs::create_dir_all(third.join("piece/sub")).unwrap();
        std::fs::write(third.join("piece/README.md"), "no licence here\n").unwrap();
        std::fs::write(third.join("piece/sub/extra.txt"), "x").unwrap();
        let item = serde_json::json!({"licence": "MIT", "upstream": "", "commit": "abc", "files": ["README.md", "gone.c"], "patch": "fix.patch"});
        std::fs::write(third.join("manifest.json"), serde_json::json!({"items": {"piece": item, "absent": {}}}).to_string()).unwrap();
        assert_eq!(
            check(&third).unwrap(),
            [
                "piece: missing 'licence_file'",
                "piece: missing 'upstream'",
                "piece: commit must be a full 40-character hash",
                "piece: listed but missing: gone.c",
                "piece: present but not listed: sub/extra.txt",
                "piece: README.md must name the licence MIT",
                "piece: patch file not listed",
                "absent: missing 'licence'",
                "absent: missing 'licence_file'",
                "absent: missing 'upstream'",
                "absent: missing 'commit'",
                "absent: missing 'files'",
                "absent: commit must be a full 40-character hash",
                "absent: README.md must name the licence None",
            ]
        );
        std::fs::remove_dir_all(&third).unwrap();
    }

    #[test]
    fn list_pads_like_the_script() {
        let items = load_manifest(&repo_third()).unwrap();
        let lines = list(&items);
        assert!(lines[0].starts_with("vkd3d-proton   LGPL-2.1  https://github.com/HansKristian-Work/vkd3d-proton @ cae8353ed795"), "{}", lines[0]);
    }

    #[test]
    fn setup_refuses_what_is_not_a_checkout() {
        let dir = scratch("setup");
        let item = serde_json::json!({"commit": "0".repeat(40), "patch": "x.patch", "upstream": "u", "licence": "L"});
        let error = setup_one(&repo_third(), "piece", &item, &dir.to_string_lossy(), true, &mut |_| {}).unwrap();
        assert_eq!(error, format!("piece: {} is not a git checkout", display(&resolve(&dir.to_string_lossy()))));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
