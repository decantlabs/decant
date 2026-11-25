//! The library side of `extract_shaders`: finds the console shader containers in a file or a folder
//! of files by their own headers, splits the FSSM block, writes the `shaders_gpu.pack` layout read back
//! by the shader research tools, and lays out the output files.

pub mod container;
pub mod fssm;
pub mod output;
pub mod pack;
pub mod parallel;
#[cfg(test)]
mod test_util;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Shaders decoded per worker task.
const BATCH: usize = 256;

/// The largest container a scan accepts, and the most parts its table may list.
const MAX_CONTAINER: usize = 64 << 20;
const MAX_PARTS: usize = 64;
/// Bits of [`Shader::position`] that hold the offset in the file, the rest holds the file's rank.
const OFFSET_BITS: u32 = 40;

/// Maps a whole file read only.
#[allow(unsafe_code)]
pub fn map_file(path: &Path) -> std::io::Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the map is read only and nothing here writes the file. If another program truncates
    // it while mapped, reads past the new end fault; the inputs are game files that nothing else
    // rewrites while a tool runs.
    unsafe { memmap2::Mmap::map(&file) }
}

/// One console shader container (DXBC with a console DXIL part) found in an input file.
#[derive(Clone)]
pub struct Shader {
    /// The file name, followed by `@0x<offset>` when the file holds more than one container or the
    /// container does not fill the file.
    pub name: String,
    /// Where the container starts in its file.
    pub offset: u64,
    /// The container's size in bytes.
    pub size: u64,
    rank: u64,
    data: Arc<dyn AsRef<[u8]> + Send + Sync>,
}

impl Shader {
    /// A shader whose container is `bytes`, not read from a file.
    pub fn from_bytes(name: impl Into<String>, bytes: Vec<u8>) -> Shader {
        Shader { name: name.into(), offset: 0, size: bytes.len() as u64, rank: 0, data: Arc::new(bytes) }
    }

    /// The container's bytes.
    pub fn container(&self) -> &[u8] {
        &(*self.data).as_ref()[self.offset as usize..(self.offset + self.size) as usize]
    }

    /// A number that identifies the container in a corpus: the rank of its file in the sorted list
    /// in the high bits, its offset in the file in the low 40 bits. For one input file it is the offset.
    pub fn position(&self) -> u64 {
        self.rank << OFFSET_BITS | self.offset
    }
}

/// The offset and size of every DXBC container in `data`, found by the container header: the magic,
/// a size that fits the data and a part table that lies inside the container.
pub fn scan_containers(data: &[u8]) -> Vec<(u64, u64)> {
    let u32_at = |at: usize| data.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(hit) = memchr::memmem::find(&data[from..], b"DXBC") {
        let start = from + hit;
        from = start + 4;
        let (Some(size), Some(parts)) = (u32_at(start + 24), u32_at(start + 28)) else { continue };
        if !(32..=MAX_CONTAINER).contains(&size) || start + size > data.len() || parts == 0 || parts > MAX_PARTS || 32 + 4 * parts > size {
            continue;
        }
        let table_ok = (0..parts).all(|i| u32_at(start + 32 + 4 * i).is_some_and(|at| at.checked_add(8).is_some_and(|end| end <= size)));
        if table_ok {
            found.push((start as u64, size as u64));
            from = start + size;
        }
    }
    found
}

/// Every shader container in `path`: one file, or every file under a folder (sorted by path). Files
/// that cannot be read or hold no container are skipped. Containers come in file order, then offset order.
pub fn find_shaders(path: &Path) -> Result<Vec<Shader>, String> {
    let mut files = Vec::new();
    if path.is_dir() {
        walk(path, &mut files).map_err(|e| format!("{}: {e}", path.display()))?;
        files.sort();
    } else if path.exists() {
        files.push(path.to_path_buf());
    } else {
        return Err(format!("{}: no such file or folder", path.display()));
    }
    let single = path.is_file();
    let mut shaders = Vec::new();
    for (rank, file) in files.iter().enumerate() {
        let Ok(data) = map_file(file) else { continue };
        let found = scan_containers(&data);
        if found.is_empty() {
            continue;
        }
        let label = if single { file.file_name().map(PathBuf::from) } else { file.strip_prefix(path).ok().map(Path::to_path_buf) }
            .unwrap_or_else(|| file.clone())
            .to_string_lossy()
            .replace('\\', "/");
        let whole_file = found.len() == 1 && found[0] == (0, data.len() as u64);
        let data: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(data);
        for (offset, size) in found {
            let name = if whole_file { label.clone() } else { format!("{label}@0x{offset:x}") };
            shaders.push(Shader { name, offset, size, rank: rank as u64, data: Arc::clone(&data) });
        }
    }
    Ok(shaders)
}

fn walk(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

/// `work(shader)` for every shader, on every core, with the results in the order of `shaders`.
pub fn map_shaders<T: Send>(shaders: &[&Shader], work: impl Fn(&Shader) -> T + Sync) -> Result<Vec<T>, String> {
    let batches = shaders.len().div_ceil(BATCH);
    Ok(parallel::map_pieces(
        batches,
        |_| 1.0 / batches.max(1) as f64,
        |b| Some(shaders[b * BATCH..shaders.len().min((b + 1) * BATCH)].iter().map(|shader| work(shader)).collect::<Vec<_>>()),
        &mut |_| {},
        &AtomicBool::new(false),
    )
    .ok_or("decoding stopped")?
    .into_iter()
    .flatten()
    .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A container with a valid header: `body` is its one part, named `DXIL`.
    fn container(body: &[u8]) -> Vec<u8> {
        let mut blob = b"DXBC".to_vec();
        blob.extend([0u8; 20]);
        blob.extend(((36 + 8 + body.len()) as u32).to_le_bytes());
        blob.extend(1u32.to_le_bytes());
        blob.extend(36u32.to_le_bytes());
        blob.extend(b"DXIL");
        blob.extend((body.len() as u32).to_le_bytes());
        blob.extend(body);
        blob
    }

    #[test]
    fn containers_are_found_by_their_header_and_not_inside_each_other() {
        let (a, b) = (container(b"first body"), container(b"second"));
        let mut data = vec![0x55u8; 7];
        data.extend(&a);
        data.extend(b"DXBC junk without a header");
        data.extend(&b);
        let found = scan_containers(&data);
        assert_eq!(found, [(7, a.len() as u64), (7 + a.len() as u64 + 26, b.len() as u64)]);
        assert!(scan_containers(b"DXBC").is_empty());
        assert!(scan_containers(&a[..a.len() - 1]).is_empty(), "size larger than the data");
    }

    #[test]
    fn a_folder_is_scanned_file_by_file() {
        let dir = std::env::temp_dir().join(format!("decant_find_shaders_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let (a, b) = (container(b"one"), container(b"two"));
        std::fs::write(dir.join("a.bin"), &a).unwrap();
        let mut two = a.clone();
        two.extend(&b);
        std::fs::write(dir.join("sub").join("b.bin"), &two).unwrap();
        std::fs::write(dir.join("c.txt"), b"nothing here").unwrap();
        let shaders = find_shaders(&dir).unwrap();
        let names: Vec<String> = shaders.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, ["a.bin".to_owned(), "sub/b.bin@0x0".to_owned(), format!("sub/b.bin@0x{:x}", a.len())]);
        assert_eq!(shaders[2].container(), &b[..]);
        assert_eq!(shaders[1].position() >> OFFSET_BITS, 2, "the rank counts every file of the sorted list");
        assert_eq!(find_shaders(&dir.join("a.bin")).unwrap()[0].name, "a.bin");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
