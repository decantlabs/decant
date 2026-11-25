//! `decant_shader_cache.py`: the translation cache. One module per pack shader, translated ahead of
//! time, so that the bridge to vkd3d-proton (`decant_shader_serve`) no longer runs the probe while the
//! game runs.
//!
//! The cache is filled from a `decant_shader_batch` measure run with `--placeholders` and with
//! `DECANT_SHADER_SOURCES=1` and `DECANT_SHADER_NO_MIP_STATS=1` in the environment. It gets `<pack
//! number>.spv` and `<pack number>.src` for each translated shader, and `cache.json`.
//! Everything in it comes from title files: keep it locally, never in the repository.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};
use decant_shader_core::py;

/// The stages a measure covers, in the order `decant_shader_batch` files them.
pub const STAGES: [u8; 5] = [0, 1, 2, 5, 13];

fn io(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// `digest(path)`: the SHA 256 of a file, in lowercase hex.
pub fn digest(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(io(path))?;
    let mut hasher = Sha256::new();
    let mut chunk = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut chunk).map_err(io(path))?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `stage_indices()`: the pack number of each shader, per stage, in the order `decant_shader_batch`
/// sorts them.
pub fn stage_indices(pack: &Path) -> Result<Vec<(u8, Vec<usize>)>, String> {
    let mut buckets: Vec<(u8, Vec<usize>)> = STAGES.iter().map(|&s| (s, Vec::new())).collect();
    let (header, _) = decant_extract_shaders::pack::read(pack, |index, _, stage| {
        if let Some((_, list)) = buckets.iter_mut().find(|(s, _)| *s == stage) {
            list.push(index);
        }
        false
    })
    .map_err(io(pack))?;
    if !header.is_current() {
        return Err(format!("AssertionError: {} is not an XSGC version 1 pack", pack.display()));
    }
    Ok(buckets)
}

/// The shader numbers of one batch whose module goes into the cache. A validated measure (with
/// `C` lines) keeps the modules `spirv-val` accepted, a measure without validation the successful
/// translations (`T` lines).
pub fn valid_locals(log: &str) -> Result<BTreeSet<i64>, String> {
    let rows: Vec<Vec<&str>> = py::splitlines(log).into_iter().map(|line| line.split('\t').collect()).collect();
    let kind = if rows.iter().any(|r| r[0] == "C") { "C" } else { "T" };
    let mut valid = BTreeSet::new();
    for r in &rows {
        if r.len() > 2 && r[0] == kind && r[2] == "OK" {
            valid.insert(py::int(r[1])?);
        }
    }
    Ok(valid)
}

/// What a fill copied: modules with their origin record, and modules left out for lack of one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Copied {
    pub copied: usize,
    pub missing_src: usize,
}

/// `seq[i]` with Python's negative indexes.
fn index(items: &[usize], i: i64) -> Result<usize, String> {
    let at = if i < 0 { i + items.len() as i64 } else { i };
    usize::try_from(at).ok().and_then(|at| items.get(at).copied()).ok_or_else(|| "IndexError: list index out of range".to_owned())
}

/// The batch folders of one stage: entries of `batch` named `<stage>-...`, sorted.
fn folders(batch: &Path, stage: u8) -> Vec<std::path::PathBuf> {
    let prefix = format!("{stage}-");
    let mut found: Vec<_> = fs::read_dir(batch)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .map(|e| e.path())
        .collect();
    found.sort_by_key(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()));
    found
}

/// Copies the accepted modules of a measure into the cache, named by their pack number.
pub fn fill(batch: &Path, cache: &Path, buckets: &[(u8, Vec<usize>)]) -> Result<Copied, String> {
    let mut counts = Copied::default();
    for (stage, indices) in buckets {
        for folder in folders(batch, *stage) {
            let name = folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let offset = py::int(name.split('-').nth(1).ok_or("IndexError: list index out of range")?)?;
            let log = folder.join("batch.tsv");
            if !log.exists() {
                continue;
            }
            let text = py::read_text_utf8(&fs::read(&log).map_err(io(&log))?);
            for local in valid_locals(&text)? {
                let (spv, src) = (folder.join(format!("{local}.spv")), folder.join(format!("{local}.src")));
                if !spv.exists() {
                    continue;
                }
                if !src.exists() {
                    counts.missing_src += 1;
                    continue;
                }
                let pack_id = index(indices, offset + local)?;
                fs::copy(&spv, cache.join(format!("{pack_id}.spv"))).map_err(io(&spv))?;
                fs::copy(&src, cache.join(format!("{pack_id}.src"))).map_err(io(&src))?;
                counts.copied += 1;
            }
        }
    }
    Ok(counts)
}

/// `len(list(cache.glob("*.spv")))`. Windows matches the suffix without regard to case.
pub fn module_count(cache: &Path) -> usize {
    fs::read_dir(cache)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if cfg!(windows) { name.to_lowercase().ends_with(".spv") } else { name.ends_with(".spv") }
        })
        .count()
}

/// The contents of `cache.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    pub modules: usize,
    pub copied: Copied,
    pub pack_sha256: String,
    pub probe_sha256: Option<String>,
}

impl Info {
    fn fields(&self) -> [(&'static str, String); 5] {
        [
            ("modules", self.modules.to_string()),
            ("copied_now", self.copied.copied.to_string()),
            ("without_origin_record", self.copied.missing_src.to_string()),
            ("pack_sha256", format!("\"{}\"", self.pack_sha256)),
            ("probe_sha256", self.probe_sha256.as_ref().map_or("null".into(), |h| format!("\"{h}\""))),
        ]
    }

    /// `json.dumps(info, indent=1)`.
    pub fn file(&self) -> String {
        let lines: Vec<String> = self.fields().iter().map(|(k, v)| format!(" \"{k}\": {v}")).collect();
        format!("{{\n{}\n}}", lines.join(",\n"))
    }

    /// `json.dumps(info)`.
    pub fn line(&self) -> String {
        let fields: Vec<String> = self.fields().iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
        format!("{{{}}}", fields.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use decant_extract_shaders::fssm::Code;
    use decant_extract_shaders::output::{pack_header, pack_record};

    fn temp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("decant-shader-cache-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn logs_pick_validated_or_translated_modules() {
        assert_eq!(valid_locals("B\t0\tx\nT\t0\tOK\t3\nC\t0\tOK\t9\nT\t1\tOK\nC\t1\tFAIL\tmsg\nT\t2\tOK").unwrap(), BTreeSet::from([0]));
        assert_eq!(valid_locals("T\t0\tOK\t3\r\nT\t1\tFAIL\r\nT\t4\tOK").unwrap(), BTreeSet::from([0, 4]));
        assert!(valid_locals("C\tx\tOK").is_err());
    }

    #[test]
    fn a_measure_fills_the_cache_by_pack_number() {
        let dir = temp("fill");
        let pack = dir.join("shaders_gpu.pack");
        let mut bytes = Vec::new();
        pack_header(&mut bytes, 5).unwrap();
        let code = Code { sub_header: [0; 16], words: vec![1], padding: 0 };
        for (name, stage) in [("a", "vertex"), ("b", "pixel"), ("c", "vertex"), ("d", "geometry"), ("e", "pixel")] {
            pack_record(&mut bytes, name, stage, &code).unwrap();
        }
        fs::write(&pack, bytes).unwrap();
        let buckets = stage_indices(&pack).unwrap();
        assert_eq!(buckets, [(0, vec![1, 4]), (1, vec![0, 2]), (2, vec![3]), (5, vec![]), (13, vec![])]);

        let batch = dir.join("measure");
        let pixel = batch.join("0-00000");
        fs::create_dir_all(&pixel).unwrap();
        fs::write(pixel.join("batch.tsv"), "T\t0\tOK\nT\t1\tOK\n").unwrap();
        for (n, text) in [("0.spv", "m0"), ("0.src", "s0"), ("1.spv", "m1")] {
            fs::write(pixel.join(n), text).unwrap();
        }
        let vertex = batch.join("1-00000");
        fs::create_dir_all(&vertex).unwrap();
        fs::write(vertex.join("batch.tsv"), "T\t1\tOK\nC\t1\tOK\n").unwrap();
        fs::write(vertex.join("1.spv"), "m2").unwrap();
        fs::write(vertex.join("1.src"), "s2").unwrap();
        let geometry = batch.join("2-00000");
        fs::create_dir_all(&geometry).unwrap();
        fs::write(geometry.join("batch.tsv"), "T	0	OK
").unwrap();
        fs::write(geometry.join("0.spv"), "m3").unwrap();
        fs::write(geometry.join("0.src"), "s3").unwrap();
        fs::create_dir_all(batch.join("13-00000")).unwrap();

        let cache = dir.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let copied = fill(&batch, &cache, &buckets).unwrap();
        assert_eq!(copied, Copied { copied: 3, missing_src: 1 });
        assert_eq!(fs::read_to_string(cache.join("1.spv")).unwrap(), "m0");
        assert_eq!(fs::read_to_string(cache.join("2.src")).unwrap(), "s2");
        assert_eq!(fs::read_to_string(cache.join("3.spv")).unwrap(), "m3");
        assert_eq!(module_count(&cache), 3);

        let info = Info { modules: 2, copied: Copied { copied: 2, missing_src: 1 }, pack_sha256: digest(&pack).unwrap(), probe_sha256: None };
        assert_eq!(info.pack_sha256.len(), 64);
        assert_eq!(
            info.line(),
            format!("{{\"modules\": 2, \"copied_now\": 2, \"without_origin_record\": 1, \"pack_sha256\": \"{}\", \"probe_sha256\": null}}", info.pack_sha256)
        );
        assert!(info.file().starts_with("{\n \"modules\": 2,\n \"copied_now\": 2,\n"));
        assert!(info.file().ends_with(" \"probe_sha256\": null\n}"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sha256_of_a_known_text() {
        let dir = temp("digest");
        let path = dir.join("abc");
        fs::write(&path, "abc").unwrap();
        assert_eq!(digest(&path).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        fs::remove_dir_all(&dir).unwrap();
    }
}
