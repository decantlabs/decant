//! Staging: the package folder stays read-only; a writable overlay folder receives a link to each
//! of its entries (a junction for a folder, a hard link or else a symbolic link for a file) and a
//! copy of every shim DLL. The title starts from the overlay, so the shims sit in its application
//! folder, the first place Windows looks for a DLL that is not a known system DLL.
//!
//! Windows has no overlay file system: a title that opens a linked file for writing writes the
//! package's file. GDK titles write to their save and local storage folders, not to their install
//! folder, but nothing enforces it here; `staged` lists what is linked.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The list of what the overlay holds, so a new staging removes exactly that and nothing else.
pub const MANIFEST: &str = ".decant_launch_stage.txt";

/// How one entry of the overlay was made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Made {
    /// A junction to a package folder.
    Junction,
    /// A real folder created for a recursively staged hardlink tree.
    RealDir,
    /// A hard link to a package file (same volume).
    HardLink,
    /// A symbolic link to a package file (another volume; needs the right to create them).
    SymLink,
    /// A copy of a shim DLL.
    Shim,
}

impl Made {
    fn word(self) -> &'static str {
        match self {
            Made::Junction => "junction",
            Made::RealDir => "realdir",
            Made::HardLink => "hardlink",
            Made::SymLink => "symlink",
            Made::Shim => "shim",
        }
    }

    fn from_word(word: &str) -> Option<Made> {
        Some(match word {
            "junction" => Made::Junction,
            "realdir" => Made::RealDir,
            "hardlink" => Made::HardLink,
            "symlink" => Made::SymLink,
            "shim" => Made::Shim,
            _ => return None,
        })
    }
}

/// One entry made in the overlay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub made: Made,
}

fn inside(path: &Path, folder: &Path) -> bool {
    path.starts_with(folder)
}

/// Checks that the package and the overlay are two separate existing folders.
fn check_folders(package: &Path, overlay: &Path) -> io::Result<(PathBuf, PathBuf)> {
    let package = fs::canonicalize(package)?;
    // Checked before anything is created: nothing is ever made inside the package.
    let overlay_abs = if overlay.exists() {
        fs::canonicalize(overlay)?
    } else {
        let parent = overlay.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let name = overlay.file_name().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the overlay needs a folder name"))?;
        fs::canonicalize(parent)?.join(name)
    };
    if inside(&overlay_abs, &package) || inside(&package, &overlay_abs) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "the overlay and the package must not contain each other"));
    }
    fs::create_dir_all(&overlay_abs)?;
    Ok((package, overlay_abs))
}

/// The entries a previous staging made, from its manifest.
pub fn staged(overlay: &Path) -> Vec<Entry> {
    let Ok(text) = fs::read_to_string(overlay.join(MANIFEST)) else { return Vec::new() };
    text.lines()
        .filter_map(|line| {
            let (word, name) = line.split_once('\t')?;
            Some(Entry { name: name.to_owned(), made: Made::from_word(word)? })
        })
        .collect()
}

/// Removes what a previous staging made: links are removed as links (a junction's folder and a
/// linked file's data stay in the package), copies are deleted. Nothing else is touched.
pub fn unstage(overlay: &Path) -> io::Result<()> {
    let mut entries = staged(overlay);
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.name.matches('/').count()));
    for entry in entries {
        let path = overlay.join(&entry.name);
        let Ok(meta) = fs::symlink_metadata(&path) else { continue };
        if ((entry.made == Made::Junction || entry.made == Made::RealDir) && meta.is_dir()) || (entry.made == Made::Junction && meta.file_type().is_symlink()) {
            // On a junction, RemoveDirectory removes the reparse point only.
            fs::remove_dir(&path)?;
        } else {
            fs::remove_file(&path)?;
        }
    }
    let manifest = overlay.join(MANIFEST);
    if manifest.exists() {
        fs::remove_file(manifest)?;
    }
    Ok(())
}

#[cfg(windows)]
fn junction(link: &Path, target: &Path) -> io::Result<()> {
    use std::os::windows::process::CommandExt;
    // mklink /J needs no privilege; raw_arg keeps cmd's own quoting. CREATE_NO_WINDOW: a window
    // app (the Decant Launcher) stages without a console flashing up for each folder.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let status = std::process::Command::new("cmd")
        .raw_arg(format!("/C mklink /J \"{}\" \"{}\" >NUL", link.display(), target.display()))
        .creation_flags(CREATE_NO_WINDOW)
        .status()?;
    if status.success() { Ok(()) } else { Err(io::Error::other(format!("mklink /J failed for {}", link.display()))) }
}

#[cfg(not(windows))]
fn junction(link: &Path, target: &Path) -> io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

fn file_link(link: &Path, target: &Path) -> io::Result<Made> {
    if fs::hard_link(target, link).is_ok() {
        return Ok(Made::HardLink);
    }
    #[cfg(windows)]
    let linked = std::os::windows::fs::symlink_file(target, link);
    #[cfg(not(windows))]
    let linked = std::os::unix::fs::symlink(target, link);
    linked.map(|()| Made::SymLink).map_err(|e| io::Error::new(e.kind(), format!("cannot link {} (other volume and no right to create symbolic links: {e})", target.display())))
}

fn link_tree_file_with(link: &Path, target: &Path, fallback: impl FnOnce(&Path, &Path) -> io::Result<Made>) -> io::Result<Made> {
    match fs::hard_link(target, link) {
        Ok(()) => Ok(Made::HardLink),
        Err(error) => {
            eprintln!("decant-launch: hard link failed for {} ({error}), using file link fallback", target.display());
            fallback(link, target)
        }
    }
}

fn link_tree_file(link: &Path, target: &Path) -> io::Result<Made> {
    link_tree_file_with(link, target, file_link)
}

fn stage_tree(package: &Path, overlay: &Path, files: &mut Vec<Entry>, prefix: &Path) -> io::Result<()> {
    for entry in fs::read_dir(package)? {
        let entry = entry?;
        let source = entry.path();
        let target = overlay.join(entry.file_name());
        let relative = prefix.join(entry.file_name());
        if source.is_dir() {
            fs::create_dir_all(&target)?;
            stage_tree(&source, &target, files, &relative)?;
        } else {
            let how = link_tree_file(&target, &source)?;
            files.push(Entry { name: relative.to_string_lossy().replace('\\', "/"), made: how });
        }
    }
    Ok(())
}

/// The console DirectStorage spec of a retail title, staged as `SPEC_NAME` for the DirectStorage shim.
const RETAIL_SPEC: &str = include_str!("../../decant-dstorage/abi/retail_title.abi");
/// The file name `dstorage_xs.dll` looks for next to itself.
pub const SPEC_NAME: &str = "decant_storage_abi.txt";

/// Stages `package` into `overlay`, then copies every `.dll` of `shims` there (in place of a package
/// file of the same name). A previous staging is removed first. Returns what was made, in order.
pub fn stage(package: &Path, overlay: &Path, shims: Option<&Path>) -> io::Result<Vec<Entry>> {
    stage_with_renderer(package, overlay, shims, None)
}

/// The two DLLs of a vkd3d-proton build: `d3d12.dll` is found first (an application folder wins
/// over System32), `d3d12core.dll` must sit next to it.
pub const VKD3D_DLLS: [&str; 2] = ["d3d12.dll", "d3d12core.dll"];

/// The lower case hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    use sha2::Digest;
    let digest = sha2::Sha256::digest(fs::read(path)?);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// Checks that `vkd3d` holds both renderer DLLs and returns their paths, in [`VKD3D_DLLS`] order.
pub fn vkd3d_files(vkd3d: &Path) -> io::Result<Vec<PathBuf>> {
    VKD3D_DLLS
        .iter()
        .map(|name| {
            let path = vkd3d.join(name);
            if path.is_file() { Ok(path) } else { Err(io::Error::new(io::ErrorKind::NotFound, format!("{name} is missing in the vkd3d folder {}", vkd3d.display()))) }
        })
        .collect()
}

/// True when the overlay holds vkd3d's `d3d12core.dll` (staged with `--vkd3d`).
pub fn has_vkd3d(overlay: &Path) -> bool {
    staged(overlay).iter().any(|e| e.made == Made::Shim && e.name.eq_ignore_ascii_case("d3d12core.dll"))
}

/// The variables a run adds when the overlay holds vkd3d: a warning level log in the overlay. A
/// variable already set (`lookup` answers) is left to its owner, and the shader bridge variables
/// (`VKD3D_DECANT_SHADER_*`, `VKD3D_SHADER_OVERRIDE`, `DECANT_SHADER_*`) are never set here: the child
/// inherits them when the caller has them.
pub fn renderer_env(overlay: &Path, lookup: impl Fn(&str) -> Option<String>) -> Vec<(String, String)> {
    let mut env = Vec::new();
    if !has_vkd3d(overlay) {
        return env;
    }
    if lookup("VKD3D_DEBUG").is_none() {
        env.push(("VKD3D_DEBUG".to_owned(), "warn".to_owned()));
    }
    if lookup("VKD3D_LOG_FILE").is_none() {
        env.push(("VKD3D_LOG_FILE".to_owned(), overlay.join("vkd3d.log").display().to_string()));
    }
    env
}

/// [`stage`], and with `vkd3d` also the renderer DLLs of that folder as shims (they replace a
/// package or shim file of the same name). Both must exist, or nothing is staged.
pub fn stage_with_renderer(package: &Path, overlay: &Path, shims: Option<&Path>, vkd3d: Option<&Path>) -> io::Result<Vec<Entry>> {
    let renderer = vkd3d.map(vkd3d_files).transpose()?.unwrap_or_default();
    let (package, overlay) = check_folders(package, overlay)?;
    unstage(&overlay)?;
    let shim_names: Vec<(String, PathBuf)> = match shims {
        Some(dir) => {
            let mut found: Vec<(String, PathBuf)> = fs::read_dir(dir)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("dll")) && p.is_file())
                .filter_map(|p| Some((p.file_name()?.to_string_lossy().into_owned(), p)))
                .collect();
            found.sort();
            found
        }
        None => Vec::new(),
    };
    let mut shim_names = shim_names;
    for path in renderer {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        shim_names.retain(|(s, _)| !s.eq_ignore_ascii_case(&name));
        shim_names.push((name, path));
    }
    shim_names.sort();
    let is_shim = |name: &str| shim_names.iter().any(|(s, _)| s.eq_ignore_ascii_case(name));
    let mut made = Vec::new();
    let hardlink_mode = std::env::var("DECANT_STAGE_MODE").is_ok_and(|mode| mode == "hardlink");
    let mut names: Vec<PathBuf> = fs::read_dir(&package)?.filter_map(Result::ok).map(|e| e.path()).collect();
    names.sort();
    for source in names {
        let Some(name) = source.file_name().map(|n| n.to_string_lossy().into_owned()) else { continue };
        if is_shim(&name) || name == MANIFEST {
            continue;
        }
        let link = overlay.join(&name);
        if fs::symlink_metadata(&link).is_ok() {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} already exists in the overlay and is not from a staging", link.display())));
        }
        let how = if source.is_dir() {
            if hardlink_mode && matches!(name.as_str(), "pack" | "data") {
                fs::create_dir_all(&link)?;
                stage_tree(&source, &link, &mut made, Path::new(&name))?;
                Made::RealDir
            } else {
                junction(&link, &source)?;
                Made::Junction
            }
        } else {
            file_link(&link, &source)?
        };
        made.push(Entry { name, made: how });
    }
    for (name, path) in &shim_names {
        fs::copy(path, overlay.join(name))?;
        made.push(Entry { name: name.clone(), made: Made::Shim });
    }
    // With shims, a title folder that brings no spec of its own gets ours (one `Shim` entry, so it is unstaged too).
    if (shims.is_some() || vkd3d.is_some()) && !package.join(SPEC_NAME).exists() {
        fs::write(overlay.join(SPEC_NAME), RETAIL_SPEC)?;
        made.push(Entry { name: SPEC_NAME.to_owned(), made: Made::Shim });
    }
    let manifest: String = made.iter().map(|e| format!("{}\t{}\n", e.made.word(), e.name)).collect();
    fs::write(overlay.join(MANIFEST), manifest)?;
    Ok(made)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package() -> (tempfile::TempDir, tempfile::TempDir, tempfile::TempDir) {
        let package = tempfile::tempdir().unwrap();
        fs::write(package.path().join("Title.exe"), b"MZ").unwrap();
        fs::write(package.path().join("MicrosoftGame.config"), b"<Game/>").unwrap();
        fs::write(package.path().join("xmem.dll"), b"package copy").unwrap();
        fs::create_dir(package.path().join("data")).unwrap();
        fs::write(package.path().join("data").join("a.dat"), b"DATA").unwrap();
        let shims = tempfile::tempdir().unwrap();
        fs::write(shims.path().join("xmem.dll"), b"our shim").unwrap();
        fs::write(shims.path().join("d3d12_xs.dll"), b"our placeholder").unwrap();
        fs::write(shims.path().join("readme.txt"), b"not a dll").unwrap();
        (package, shims, tempfile::tempdir().unwrap())
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!("decant-stage-{label}-{}-{}", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn hardlink_tree_has_real_directories_and_same_file_identity() {
        let root = unique_test_dir("tree");
        let package = root.join("package");
        let overlay = root.join("overlay");
        fs::create_dir_all(package.join("pack/nested/deep")).unwrap();
        fs::create_dir_all(package.join("data/config")).unwrap();
        fs::write(package.join("pack/nested/deep/archive.bin"), b"archive").unwrap();
        fs::write(package.join("data/config/settings.bin"), b"settings").unwrap();
        fs::create_dir_all(&overlay).unwrap();
        let mut files = Vec::new();
        fs::create_dir_all(overlay.join("pack")).unwrap();
        fs::create_dir_all(overlay.join("data")).unwrap();
        stage_tree(&package.join("pack"), &overlay.join("pack"), &mut files, Path::new("pack")).unwrap();
        stage_tree(&package.join("data"), &overlay.join("data"), &mut files, Path::new("data")).unwrap();
        let source = package.join("pack/nested/deep/archive.bin");
        let linked = overlay.join("pack/nested/deep/archive.bin");
        assert!(overlay.join("pack/nested/deep").is_dir());
        assert!(!fs::symlink_metadata(overlay.join("pack/nested")).unwrap().file_type().is_symlink());
        assert_eq!(fs::read(&linked).unwrap(), b"archive");
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let src = fs::metadata(&source).unwrap();
            let dst = fs::metadata(&linked).unwrap();
            assert_eq!((src.dev(), src.ino()), (dst.dev(), dst.ino()));
        }
        #[cfg(windows)]
        {
            let identity = |path: &Path| {
                let output = std::process::Command::new("fsutil").args(["file", "queryfileid", &path.display().to_string()]).output().unwrap();
                assert!(output.status.success(), "fsutil file identity query failed");
                String::from_utf8_lossy(&output.stdout).trim().to_owned()
            };
            assert_eq!(identity(&source), identity(&linked));
        }
        assert_eq!(files.iter().map(|entry| entry.made).collect::<Vec<_>>(), [Made::HardLink, Made::HardLink]);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn failed_hardlink_uses_file_link_fallback() {
        let root = unique_test_dir("fallback");
        let source = root.join("source");
        let target = root.join("target");
        fs::write(&source, b"fallback content").unwrap();
        fs::create_dir(&target).unwrap();
        let fallback_path = root.join("fallback");
        let made = link_tree_file_with(&target, &source, |_, source| {
            fs::copy(source, &fallback_path)?;
            Ok(Made::Shim)
        }).unwrap();
        assert_eq!(made, Made::Shim);
        assert_eq!(fs::read(&fallback_path).unwrap(), b"fallback content");
        fs::remove_file(fallback_path).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_overlay_links_the_package_and_puts_the_shims_first() {
        let (package, shims, overlay) = package();
        let made = stage(package.path(), overlay.path(), Some(shims.path())).unwrap();
        let summary: Vec<(&str, Made)> = made.iter().map(|e| (e.name.as_str(), e.made)).collect();
        assert_eq!(summary, [("MicrosoftGame.config", Made::HardLink), ("Title.exe", Made::HardLink), ("data", Made::Junction), ("d3d12_xs.dll", Made::Shim), ("xmem.dll", Made::Shim), (SPEC_NAME, Made::Shim)]);
        assert_eq!(fs::read_to_string(overlay.path().join(SPEC_NAME)).unwrap(), RETAIL_SPEC);
        let o = overlay.path();
        assert_eq!(fs::read(o.join("xmem.dll")).unwrap(), b"our shim", "the shim, not the package's copy");
        assert_eq!(fs::read(o.join("data").join("a.dat")).unwrap(), b"DATA");
        assert!(!o.join("readme.txt").exists());
        assert_eq!(staged(o), made);
    }

    #[test]
    fn a_spec_of_the_title_folder_is_kept_and_without_shims_none_is_added() {
        let (package, shims, overlay) = package();
        fs::write(package.path().join(SPEC_NAME), b"abi 1
").unwrap();
        let made = stage(package.path(), overlay.path(), Some(shims.path())).unwrap();
        assert_eq!(made.iter().filter(|e| e.name == SPEC_NAME).map(|e| e.made).collect::<Vec<_>>(), [Made::HardLink], "the title's own spec is linked, not replaced");
        fs::remove_file(package.path().join(SPEC_NAME)).unwrap();
        stage(package.path(), overlay.path(), None).unwrap();
        assert!(!overlay.path().join(SPEC_NAME).exists());
        stage(package.path(), overlay.path(), Some(shims.path())).unwrap();
        unstage(overlay.path()).unwrap();
        assert!(!overlay.path().join(SPEC_NAME).exists(), "unstaging removes the spec");
    }

    #[test]
    fn staging_again_removes_links_without_touching_the_package() {
        let (package, shims, overlay) = package();
        stage(package.path(), overlay.path(), Some(shims.path())).unwrap();
        stage(package.path(), overlay.path(), None).unwrap();
        assert_eq!(fs::read(overlay.path().join("xmem.dll")).unwrap(), b"package copy", "without shims the package's own file is linked");
        unstage(overlay.path()).unwrap();
        assert_eq!(fs::read_dir(overlay.path()).unwrap().count(), 0, "the overlay is empty again");
        assert_eq!(fs::read(package.path().join("data").join("a.dat")).unwrap(), b"DATA", "the junction's target is intact");
        assert_eq!(fs::read(package.path().join("Title.exe")).unwrap(), b"MZ");
    }

    fn vkd3d_folder() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("d3d12.dll"), b"vkd3d d3d12").unwrap();
        fs::write(dir.path().join("d3d12core.dll"), b"vkd3d core").unwrap();
        dir
    }

    #[test]
    fn the_renderer_dlls_are_staged_as_shims_and_unstaged() {
        let (package, shims, overlay) = package();
        fs::write(package.path().join("d3d12core.dll"), b"package core").unwrap();
        let vk = vkd3d_folder();
        let made = stage_with_renderer(package.path(), overlay.path(), Some(shims.path()), Some(vk.path())).unwrap();
        let o = overlay.path();
        assert_eq!(fs::read(o.join("d3d12.dll")).unwrap(), b"vkd3d d3d12");
        assert_eq!(fs::read(o.join("d3d12core.dll")).unwrap(), b"vkd3d core", "not the package's copy");
        assert!(made.iter().any(|e| e.name == "d3d12.dll" && e.made == Made::Shim));
        assert!(has_vkd3d(o));
        unstage(o).unwrap();
        assert_eq!(fs::read_dir(o).unwrap().count(), 0);
        assert!(!has_vkd3d(o));
        assert_eq!(fs::read(package.path().join("d3d12core.dll")).unwrap(), b"package core");
    }

    #[test]
    fn a_renderer_folder_without_both_dlls_stages_nothing() {
        let (package, _shims, overlay) = package();
        let vk = vkd3d_folder();
        fs::remove_file(vk.path().join("d3d12core.dll")).unwrap();
        let err = stage_with_renderer(package.path(), overlay.path(), None, Some(vk.path())).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("d3d12core.dll"));
        assert_eq!(fs::read_dir(overlay.path()).unwrap().count(), 0);
        stage(package.path(), overlay.path(), None).unwrap();
        assert!(!has_vkd3d(overlay.path()), "without the folder nothing changes");
    }

    #[test]
    fn the_sha256_is_the_known_digest() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a"), b"abc").unwrap();
        assert_eq!(sha256_file(&dir.path().join("a")).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn the_renderer_env_sets_the_log_only_when_vkd3d_is_staged_and_not_set() {
        let (package, _shims, overlay) = package();
        let none = |_: &str| None;
        assert!(renderer_env(overlay.path(), none).is_empty());
        let vk = vkd3d_folder();
        stage_with_renderer(package.path(), overlay.path(), None, Some(vk.path())).unwrap();
        let env = renderer_env(overlay.path(), none);
        assert_eq!(env[0], ("VKD3D_DEBUG".to_owned(), "warn".to_owned()));
        assert_eq!(env[1].0, "VKD3D_LOG_FILE");
        assert!(env[1].1.ends_with("vkd3d.log"));
        assert_eq!(env.len(), 2, "no bridge variable is invented");
        let set = |k: &str| (k == "VKD3D_DEBUG").then(|| "trace".to_owned());
        let env = renderer_env(overlay.path(), set);
        assert_eq!(env.len(), 1);
        assert_eq!(env[0].0, "VKD3D_LOG_FILE");
        let both = |_: &str| Some("x".to_owned());
        assert!(renderer_env(overlay.path(), both).is_empty());
    }

    #[test]
    fn foreign_files_and_nested_folders_are_refused() {
        let (package, _shims, overlay) = package();
        fs::write(overlay.path().join("Title.exe"), b"someone's file").unwrap();
        assert_eq!(stage(package.path(), overlay.path(), None).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(overlay.path().join("Title.exe")).unwrap(), b"someone's file", "never overwritten");
        let nested = package.path().join("overlay");
        assert_eq!(stage(package.path(), &nested, None).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert!(!nested.exists(), "nothing is created inside the package");
    }
}
