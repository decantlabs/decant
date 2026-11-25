//! Collecting the imports of every PE file and checking each function against the package and
//! this PC.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::host::{Checker, Host};
use crate::pe::{Func, Pe, SectionRow};
use crate::pyfmt;

pub struct FileInfo {
    pub size_mib: f64,
    pub machine: u16,
    pub dll_count: usize,
    pub func_count: usize,
    pub sections: Option<Vec<SectionRow>>,
}

pub struct Dll {
    /// The name as first seen.
    pub name: String,
    pub origin: &'static str,
    pub users: BTreeSet<String>,
    /// Functions in first seen order, with whether they resolve.
    pub funcs: Vec<(Func, bool)>,
    pub delay: bool,
}

pub struct Audit {
    pub root: PathBuf,
    /// Relative path and facts, in the order the files were read.
    pub files: Vec<(String, FileInfo)>,
    pub errors: Vec<(String, String)>,
    /// Lowercase name and entry, in first seen order.
    pub dlls: Vec<(String, Dll)>,
    /// Whether a folder of our own DLLs (`--shims`) was given, so that origin is reported.
    pub shims: bool,
}

fn is_pe_name(path: &Path) -> bool {
    path.extension().is_some_and(|e| matches!(e.to_string_lossy().to_lowercase().as_str(), "exe" | "dll"))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(t) if t.is_dir() => walk(&path, out),
            Ok(t) if t.is_file() => out.push(path),
            _ => {}
        }
    }
}

fn in_work_folder(path: &Path, root: &Path) -> bool {
    path.strip_prefix(root).ok().and_then(Path::parent).is_some_and(|p| p.components().any(|c| c.as_os_str().to_string_lossy().starts_with("01_")))
}

/// The `.exe` and `.dll` files to audit and the folder they are reported against.
pub fn targets(target: &Path) -> (PathBuf, Vec<PathBuf>) {
    let target = pyfmt::normalize(target);
    if target.is_dir() {
        let mut files = Vec::new();
        walk(&target, &mut files);
        files.retain(|p| is_pe_name(p) && !in_work_folder(p, &target));
        // Python sorts Windows paths without regard to case.
        files.sort_by_key(|p| pyfmt::text(p).to_lowercase());
        (target, files)
    } else {
        let root = target.parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        (pyfmt::normalize(&root), vec![target])
    }
}

fn relative(path: &Path, root: &Path) -> String {
    pyfmt::text(path.strip_prefix(root).unwrap_or(path))
}

/// The DLLs of a folder of our own compatibility DLLs, by lowercase file name (read as data).
fn load_shims(folder: &Path) -> HashMap<String, Pe> {
    let mut out = HashMap::new();
    for entry in std::fs::read_dir(folder).into_iter().flatten().flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
        if name.ends_with(".dll")
            && let Ok(pe) = Pe::open(&path)
        {
            out.insert(name, pe);
        }
    }
    out
}

pub fn run(target: &Path, files: Vec<PathBuf>, root: PathBuf, host: &mut Host, shim_folder: Option<&Path>) -> Audit {
    let shims = shim_folder.map(load_shims).unwrap_or_default();
    let mut parsed: Vec<(PathBuf, Pe)> = Vec::new();
    let mut package: HashMap<String, usize> = HashMap::new();
    let mut siblings: Vec<Pe> = Vec::new();
    let mut errors = Vec::new();
    for path in files {
        match Pe::open(&path) {
            Ok(pe) => {
                package.insert(path.file_name().unwrap_or_default().to_string_lossy().to_lowercase(), parsed.len());
                parsed.push((path, pe));
            }
            Err(e) => errors.push((relative(&path, &root), e)),
        }
    }
    let mut sibling_index: HashMap<String, usize> = HashMap::new();
    if !pyfmt::normalize(target).is_dir() {
        for entry in std::fs::read_dir(&root).into_iter().flatten().flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            if name.ends_with(".dll")
                && !package.contains_key(&name)
                && !sibling_index.contains_key(&name)
                && let Ok(pe) = Pe::open(&path)
            {
                sibling_index.insert(name, siblings.len());
                siblings.push(pe);
            }
        }
    }

    let mut files_out = Vec::new();
    let mut dlls: Vec<(String, Dll)> = Vec::new();
    let mut by_key: HashMap<String, usize> = HashMap::new();
    let mut seen_funcs: Vec<HashSet<Func>> = Vec::new();
    for (path, pe) in &parsed {
        let rel = relative(path, &root);
        let imports = pe.imports();
        let distinct: BTreeSet<String> = imports.iter().map(|i| i.dll.to_lowercase()).collect();
        let is_exe = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exe"));
        let size = std::fs::metadata(path).map_or(0, |m| m.len());
        files_out.push((
            rel.clone(),
            FileInfo {
                size_mib: size as f64 / 1_048_576.0,
                machine: pe.machine,
                dll_count: distinct.len(),
                func_count: imports.iter().map(|i| i.funcs.len()).sum(),
                sections: is_exe.then(|| pe.section_report()),
            },
        ));
        for imp in imports {
            let key = imp.dll.to_lowercase();
            let slot = *by_key.entry(key.clone()).or_insert_with(|| {
                dlls.push((key.clone(), Dll { name: imp.dll.clone(), origin: "", users: BTreeSet::new(), funcs: Vec::new(), delay: false }));
                seen_funcs.push(HashSet::new());
                dlls.len() - 1
            });
            let entry = &mut dlls[slot].1;
            entry.users.insert(rel.clone());
            entry.delay |= imp.delay;
            let seen = &mut seen_funcs[slot];
            for func in imp.funcs {
                if seen.insert(func.clone()) {
                    entry.funcs.push((func, false));
                }
            }
        }
    }

    for (key, entry) in &mut dlls {
        let local = package.get(key).map(|&i| &parsed[i].1).or_else(|| sibling_index.get(key).map(|&i| &siblings[i]));
        let checker = match local {
            Some(pe) => {
                entry.origin = "paquet";
                let (names, ordinals) = pe.exports();
                Some(Checker::Exports(names, ordinals))
            }
            None => match shims.get(key) {
                // A DLL of ours staged in the title folder is found by the Windows loader before System32.
                Some(pe) => {
                    entry.origin = "shim";
                    let (names, ordinals) = pe.exports();
                    Some(Checker::Exports(names, ordinals))
                }
                None => {
                    let found = host.lookup(&entry.name);
                    entry.origin = found.origin;
                    found.checker
                }
            },
        };
        for (func, ok) in &mut entry.funcs {
            *ok = checker.as_ref().is_some_and(|c| c.has(func));
        }
    }
    Audit { root, files: files_out, errors, dlls, shims: shim_folder.is_some() }
}
