//! Measurement folders read the way the scripts read them: `sorted(Path(folder).iterdir())` for
//! the batch folders, `batch.glob("*.spv")` for the modules of a batch.

use std::path::{Path, PathBuf};

fn io(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// `str(path)`: on Windows with backslashes, as `pathlib` writes it.
pub fn py_str(path: &Path) -> String {
    let text = path.to_string_lossy().into_owned();
    if cfg!(windows) { text.replace('/', "\\") } else { text }
}

/// The batch folders of a measurement folder. `pathlib` sorts paths case insensitively on
/// Windows, so the names are compared in lower case there.
pub fn batches(folder: &Path) -> Result<Vec<PathBuf>, String> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(folder).map_err(io(folder))?.map(|e| e.map(|e| e.path())).collect::<Result<_, _>>().map_err(io(folder))?;
    let key = |p: &PathBuf| {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if cfg!(windows) { name.to_lowercase() } else { name }
    };
    entries.sort_by_key(key);
    Ok(entries.into_iter().filter(|p| p.is_dir()).collect())
}

/// The `*.spv` entries of a batch folder in the order the directory lists them, which is the
/// order `glob` gives: hidden names are left out, and the suffix matches in any case on Windows.
pub fn modules(batch: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(batch).map_err(io(batch))? {
        let path = entry.map_err(io(batch))?.path();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let name = if cfg!(windows) { name.to_lowercase() } else { name };
        if !name.starts_with('.') && name.ends_with(".spv") {
            out.push(path);
        }
    }
    Ok(out)
}

/// The little endian words of a module, or the `struct.error` of a size that is not whole words.
pub fn words(path: &Path) -> Result<Vec<u32>, String> {
    let data = std::fs::read(path).map_err(io(path))?;
    if !data.len().is_multiple_of(4) {
        return Err(format!("struct.error: {} is not a whole number of words", path.display()));
    }
    Ok(data.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect())
}
