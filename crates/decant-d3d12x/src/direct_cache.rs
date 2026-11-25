//! The disk cache of the direct shader path (issue #172): DXIL compiled from translated HLSL and
//! the stage keys of each pipeline, keyed by everything that decides them.
//!
//! A stage key hashes the console bytes of the stage (or the translator's own HLSL key), the root
//! signature layout, the pipeline linkage, the translator version and the DXC version. A pipeline
//! key hashes the console bytes of every stage in order, the root signature layout, the linkage and
//! both versions before translation runs. On a warm bind the pipeline file gives the stage keys and
//! each stage file gives its DXIL, so a hit costs file reads only and neither translates nor
//! compiles.
//!
//! `DECANT_D3D12X_DIRECT_CACHE=<dir>` names the folder. Default: `%LOCALAPPDATA%\Decant\direct-dxil`
//! beside the run data on Windows (or `$XDG_CACHE_HOME`/`$HOME/.cache` elsewhere), never inside the
//! repository. Relative overrides are resolved under the default cache directory.
//! `DECANT_D3D12X_DIRECT_CACHE=0` switches the cache off. The cache warns when stage and pipeline
//! files exceed 2 GiB.
//!
//! Stage files are `<key as 32 lower case hex digits>.dxil` with a 48 byte header (magic `SXDC`,
//! format version, the key, the DXIL byte length and the 128 bit FNV 1a of the DXIL). Pipeline
//! files are `<key as 32 lower case hex digits>.pipe` with a 56 byte header (magic `SXDP`, format
//! version, the key, the stage count, the immediate sampler registry length and a 128 bit FNV 1a
//! checksum) followed by the stage keys. Any file whose header, version, length, sampler registry
//! prefix or payload checksum does not match is ignored and rebuilt. Files are written to a unique
//! temporary name and renamed so concurrent processes never read half a file.

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::rts0::{RegisterClass, RootLayout, RootParamKind};
use crate::shader_source::{Linkage, Pipeline, StageKind};

/// FNV 1a with 128 bits: stable across runs, builds and Rust releases, unlike the standard hasher.
#[derive(Clone, Copy, Debug)]
pub struct Fnv128(u128);

const FNV128_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV128_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

impl Default for Fnv128 {
    fn default() -> Self {
        Fnv128(FNV128_OFFSET)
    }
}

impl Fnv128 {
    /// Adds `bytes`.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        for b in bytes {
            self.0 ^= u128::from(*b);
            self.0 = self.0.wrapping_mul(FNV128_PRIME);
        }
        self
    }

    /// Adds a length prefixed field, so that `("ab", "c")` and `("a", "bc")` differ.
    pub fn field(&mut self, bytes: &[u8]) -> &mut Self {
        self.bytes(&(bytes.len() as u64).to_le_bytes()).bytes(bytes)
    }

    fn u32(&mut self, v: u32) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    /// The hash so far.
    pub fn finish(&self) -> u128 {
        self.0
    }
}

/// Bumped when the layout of the keys or of the files changes.
const FORMAT: u32 = 3;
const TRANSLATOR_HASH: &str = env!("DECANT_DIRECT_TRANSLATOR_HASH");

fn class_tag(c: RegisterClass) -> u8 {
    match c {
        RegisterClass::Srv => 0,
        RegisterClass::Uav => 1,
        RegisterClass::Cbv => 2,
        RegisterClass::Sampler => 3,
    }
}

fn hash_root(h: &mut Fnv128, root: &RootLayout) {
    h.u32(root.version).u32(root.flags).u64(root.params.len() as u64);
    for p in &root.params {
        h.u32(p.visibility);
        match &p.kind {
            RootParamKind::Table(ranges) => {
                h.bytes(&[0]).u64(ranges.len() as u64);
                for r in ranges {
                    h.bytes(&[class_tag(r.class)])
                        .u32(r.count)
                        .u32(r.base)
                        .u32(r.space)
                        .u32(r.flags)
                        .u32(r.offset);
                }
            }
            RootParamKind::Constants { register, space, count } => {
                h.bytes(&[1]).u32(*register).u32(*space).u32(*count);
            }
            RootParamKind::Descriptor { class, register, space } => {
                h.bytes(&[2, class_tag(*class)]).u32(*register).u32(*space);
            }
        }
    }
    h.u64(root.static_samplers.len() as u64);
    for s in &root.static_samplers {
        h.u32(s.register).u32(s.space).u32(s.visibility);
    }
}

fn hash_linkage(h: &mut Fnv128, linkage: &Linkage) {
    h.u64(linkage.stages.len() as u64);
    for st in &linkage.stages {
        h.field(st.short().as_bytes());
    }
    h.u64(linkage.inputs.len() as u64);
    for e in &linkage.inputs {
        h.field(e.semantic.as_bytes())
            .u32(e.index)
            .u32(e.format)
            .u32(e.slot)
            .u32(e.offset)
            .bytes(&[u8::from(e.per_instance)])
            .u32(e.step_rate);
    }
    h.u64(linkage.targets.len() as u64);
    for t in &linkage.targets {
        h.u32(*t);
    }
    h.u32(linkage.depth_format)
        .u32(linkage.topology_type)
        .u32(linkage.sample_count)
        .bytes(&[u8::from(linkage.stream_output)]);
}

/// The cache key of one stage from its console input, root layout and linkage.
pub fn stage_key(kind: StageKind, bytes: &[u8], root: &RootLayout, linkage: &Linkage, translator: &str, dxc: &str) -> u128 {
    let mut h = Fnv128::default();
    h.field(b"decant direct stage")
        .field(&FORMAT.to_le_bytes())
        .field(TRANSLATOR_HASH.as_bytes())
        .field(kind.short().as_bytes())
        .field(bytes);
    hash_root(&mut h, root);
    hash_linkage(&mut h, linkage);
    h.field(translator.as_bytes()).field(dxc.as_bytes());
    h.finish()
}

/// The cache key of one stage from the key its translator gives for it
/// (`ShaderSource::stage_keys`), which stands for the console bytes, the root signature and the
/// linkage of [`stage_key`].
pub fn source_stage_key(kind: StageKind, source_key: u128, translator: &str, dxc: &str) -> u128 {
    let mut h = Fnv128::default();
    h.field(b"decant direct stage by source key")
        .field(&FORMAT.to_le_bytes())
        .field(TRANSLATOR_HASH.as_bytes())
        .field(kind.short().as_bytes())
        .field(&source_key.to_le_bytes())
        .field(translator.as_bytes())
        .field(dxc.as_bytes());
    h.finish()
}

/// The cache key of a whole pipeline from its inputs before translation runs.
pub fn pipeline_key(pipeline: &Pipeline, translator: &str, dxc: &str) -> u128 {
    let mut h = Fnv128::default();
    h.field(b"decant direct pipeline")
        .field(&FORMAT.to_le_bytes())
        .field(TRANSLATOR_HASH.as_bytes())
        .u64(pipeline.stages.len() as u64);
    for st in &pipeline.stages {
        h.field(st.kind.short().as_bytes()).field(&st.bytes);
    }
    hash_root(&mut h, &pipeline.root);
    hash_linkage(&mut h, &pipeline.linkage);
    h.field(translator.as_bytes()).field(dxc.as_bytes());
    h.finish()
}

const STAGE_MAGIC: &[u8; 4] = b"SXDC";
const STAGE_HEADER: usize = 48;
const PIPE_MAGIC: &[u8; 4] = b"SXDP";
const PIPE_HEADER: usize = 56;
const MAX_STAGES: usize = 8;
const SAMPLERS_FILE: &str = "immediate-samplers.bin";
const MAX_SAMPLER_BYTES: usize = 8 + 16 * crate::sampler_registry::CAPACITY as usize;
const MAX_DXIL_BYTES: usize = 64 * 1024 * 1024;
const CACHE_BUDGET: u64 = 2 * 1024 * 1024 * 1024;

/// The folder of compiled DXIL and pipeline records.
#[derive(Clone, Debug)]
pub struct DiskCache {
    dir: PathBuf,
    pipeline_records: bool,
    // The estimate includes new writes. Rescan when it reaches the budget.
    size: Arc<Mutex<CacheSize>>,
}

#[derive(Debug, Default)]
struct CacheSize {
    estimate: Option<u64>,
    warned: bool,
}

static TEMP_NUMBER: AtomicU64 = AtomicU64::new(0);

impl DiskCache {
    /// A cache in `dir`, which is made when the first file is written.
    pub fn new(dir: impl Into<PathBuf>) -> DiskCache {
        let mut dir = dir.into();
        let override_value = std::env::var_os("DECANT_D3D12X_DIRECT_CACHE");
        let override_path = override_value.as_deref().filter(|value| !value.is_empty()).map(Path::new);
        let configured = Self::resolve_dir(override_value.clone(), default_dir());
        let pipeline_records = if let Some(configured) = configured.filter(|path| {
            override_path == Some(dir.as_path())
                || std::path::absolute(&dir).is_ok_and(|absolute| absolute == *path)
                || std::fs::canonicalize(&dir).ok().zip(std::fs::canonicalize(path).ok()).is_some_and(|(a, b)| a == b)
        }) {
            dir = configured;
            owns_sampler_registry(&dir)
        } else {
            // Without an override, standalone sources can use an explicit cache directory.
            override_path.is_none()
        };
        DiskCache { dir, pipeline_records, size: Arc::default() }
    }

    /// The folder.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Resolves the configured cache directory without claiming its sampler registry.
    pub fn configured_dir() -> Option<PathBuf> {
        Self::resolve_dir(std::env::var_os("DECANT_D3D12X_DIRECT_CACHE"), default_dir())
    }

    /// The configured cache if this process can own its persistent sampler registry.
    pub fn from_env() -> Option<DiskCache> {
        let dir = Self::configured_dir()?;
        owns_sampler_registry(&dir).then(|| DiskCache { dir, pipeline_records: true, size: Arc::default() })
    }

    fn resolve_dir(configured: Option<OsString>, default: Option<PathBuf>) -> Option<PathBuf> {
        let configured = non_empty(configured);
        match configured.as_deref() {
            Some(path) if path == Path::new("0") => None,
            Some(path) if path.is_absolute() => configured,
            Some(path) if path.components().all(|part| matches!(part, std::path::Component::Normal(_) | std::path::Component::CurDir)) => {
                default.filter(|base| base.is_absolute()).map(|base| base.join(path))
            }
            Some(_) => None,
            None => default.filter(|base| base.is_absolute()),
        }
    }

    fn path(&self, key: u128) -> PathBuf {
        self.dir.join(format!("{key:032x}.dxil"))
    }

    fn pipe_path(&self, key: u128) -> PathBuf {
        self.dir.join(format!("{key:032x}.pipe"))
    }

    fn samplers_state(&self) -> io::Result<(usize, u128)> {
        let mut file = match File::open(self.dir.join(SAMPLERS_FILE)) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((0, FNV128_OFFSET)),
            Err(e) => return Err(e),
        };
        let mut bytes = [0; MAX_SAMPLER_BYTES];
        let len = file.metadata()?.len();
        if len < 8 || len > bytes.len() as u64 || !(len - 8).is_multiple_of(16) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid sampler registry length"));
        }
        let len = len as usize;
        file.read_exact(&mut bytes[..len])?;
        if &bytes[..8] != b"SXSM\x01\x00\x00\x00" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid sampler registry header"));
        }
        Ok((len, Fnv128::default().bytes(&bytes[..len]).finish()))
    }

    fn samplers_prefix_hash(&self, len: usize) -> Option<u128> {
        if len == 0 {
            return Some(FNV128_OFFSET);
        }
        let mut bytes = [0; MAX_SAMPLER_BYTES];
        let prefix = bytes.get_mut(..len)?;
        File::open(self.dir.join(SAMPLERS_FILE)).ok()?.read_exact(prefix).ok()?;
        Some(Fnv128::default().bytes(prefix).finish())
    }

    /// The DXIL stored for `key`, `None` on a miss or a file that fails any header or payload check.
    pub fn read(&self, key: u128) -> Option<Vec<u8>> {
        let mut file = File::open(self.path(key)).ok()?;
        let payload_len = file.metadata().ok()?.len().checked_sub(STAGE_HEADER as u64)?;
        if !(1..=MAX_DXIL_BYTES as u64).contains(&payload_len) {
            return None;
        }
        let mut header = [0; STAGE_HEADER];
        file.read_exact(&mut header).ok()?;
        if &header[..4] != STAGE_MAGIC
            || header[4..8] != FORMAT.to_le_bytes()
            || header[8..24] != key.to_le_bytes()
            || header[24..32] != payload_len.to_le_bytes()
        {
            return None;
        }
        let mut dxil = vec![0; payload_len as usize];
        file.read_exact(&mut dxil).ok()?;
        let checksum = Fnv128::default().bytes(&dxil).finish();
        (header[32..48] == checksum.to_le_bytes()).then_some(dxil)
    }

    /// Stores `dxil` for `key`. Errors are returned for the log, the cache is best effort.
    pub fn write(&self, key: u128, dxil: &[u8]) -> std::io::Result<()> {
        if !(1..=MAX_DXIL_BYTES).contains(&dxil.len()) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid DXIL blob length"));
        }
        let checksum = Fnv128::default().bytes(dxil).finish();
        let mut file = Vec::with_capacity(STAGE_HEADER + dxil.len());
        file.extend_from_slice(STAGE_MAGIC);
        file.extend_from_slice(&FORMAT.to_le_bytes());
        file.extend_from_slice(&key.to_le_bytes());
        file.extend_from_slice(&(dxil.len() as u64).to_le_bytes());
        file.extend_from_slice(&checksum.to_le_bytes());
        file.extend_from_slice(dxil);
        let stem = format!("{key:032x}.dxil");
        let dest = self.path(key);
        self.write_atomic(&stem, &dest, &file, || self.read(key).is_some())
    }

    /// The stage keys stored for pipeline `key`, `None` on a miss or when the entry or the sampler
    /// registry prefix no longer matches.
    pub fn read_pipeline(&self, key: u128) -> Option<Vec<u128>> {
        if !self.pipeline_records {
            return None;
        }
        let mut entry = File::open(self.pipe_path(key)).ok()?;
        let len = usize::try_from(entry.metadata().ok()?.len()).ok()?;
        let mut buffer = [0; PIPE_HEADER + MAX_STAGES * 16];
        let file = buffer.get_mut(..len)?;
        if len < PIPE_HEADER {
            return None;
        }
        entry.read_exact(file).ok()?;
        if &file[..4] != PIPE_MAGIC
            || file[4..8] != FORMAT.to_le_bytes()
            || file[8..24] != key.to_le_bytes()
        {
            return None;
        }
        let count = usize::try_from(u64::from_le_bytes(file[24..32].try_into().ok()?)).ok()?;
        if !(1..=MAX_STAGES).contains(&count) || file.len() - PIPE_HEADER != count * 16 {
            return None;
        }
        let samplers_len = usize::try_from(u64::from_le_bytes(file[32..40].try_into().ok()?)).ok()?;
        let samplers_hash = self.samplers_prefix_hash(samplers_len)?;
        let checksum = u128::from_le_bytes(file[40..56].try_into().ok()?);
        let body = &file[PIPE_HEADER..];
        let want = Fnv128::default().bytes(&samplers_hash.to_le_bytes()).bytes(body).finish();
        if checksum != want {
            return None;
        }
        Some(body.as_chunks::<16>().0.iter().copied().map(u128::from_le_bytes).collect())
    }

    /// Stores `stage_keys` for pipeline `key` alongside the current immediate sampler registry state.
    pub fn write_pipeline(&self, key: u128, stage_keys: &[u128]) -> std::io::Result<()> {
        if !(1..=MAX_STAGES).contains(&stage_keys.len()) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad stage count"));
        }
        if !self.pipeline_records {
            return Ok(());
        }
        let (samplers_len, samplers_hash) = self.samplers_state()?;
        let mut body = Vec::with_capacity(stage_keys.len() * 16);
        for k in stage_keys {
            body.extend_from_slice(&k.to_le_bytes());
        }
        let checksum = Fnv128::default().bytes(&samplers_hash.to_le_bytes()).bytes(&body).finish();
        let mut file = Vec::with_capacity(PIPE_HEADER + body.len());
        file.extend_from_slice(PIPE_MAGIC);
        file.extend_from_slice(&FORMAT.to_le_bytes());
        file.extend_from_slice(&key.to_le_bytes());
        file.extend_from_slice(&(stage_keys.len() as u64).to_le_bytes());
        file.extend_from_slice(&(samplers_len as u64).to_le_bytes());
        file.extend_from_slice(&checksum.to_le_bytes());
        file.extend_from_slice(&body);
        let stem = format!("{key:032x}.pipe");
        let dest = self.pipe_path(key);
        self.write_atomic(&stem, &dest, &file, || self.read_pipeline(key).is_some())
    }

    fn write_atomic(&self, stem: &str, dest: &Path, data: &[u8], valid: impl Fn() -> bool) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let seq = TEMP_NUMBER.fetch_add(1, Ordering::Relaxed);
        let temp = self.dir.join(format!(".{stem}.{}.{nanos}.{seq}.tmp", std::process::id()));
        let mut file = File::create_new(&temp)?;
        let result = file.write_all(data);
        drop(file);
        let result = result.and_then(|()| Self::publish(|| std::fs::rename(&temp, dest), valid));
        let _ = std::fs::remove_file(&temp);
        if let Ok(true) = result {
            self.check_size(data.len() as u64, CACHE_BUDGET, |bytes| {
                #[cfg(not(windows))]
                let _ = bytes;
                #[cfg(windows)]
                crate::log::xs_log!("direct shaders: disk cache exceeds its {CACHE_BUDGET} byte budget ({bytes} bytes in {})", self.dir.display());
            });
        }
        result.map(|_| ())
    }

    fn publish(rename: impl Fn() -> io::Result<()>, valid: impl Fn() -> bool) -> io::Result<bool> {
        // Replace atomically. An open reader on Windows can briefly block replacement.
        let mut retries = 5;
        loop {
            match rename() {
                Ok(()) => return Ok(true),
                Err(_) if valid() => return Ok(false),
                Err(err) if retries == 0 => return Err(err),
                Err(_) => {
                    retries -= 1;
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
        }
    }

    fn check_size(&self, written: u64, budget: u64, warn: impl FnOnce(u64)) {
        let mut size = self.size.lock().unwrap_or_else(|e| e.into_inner());
        if size.warned {
            return;
        }
        size.estimate = size.estimate.map(|n| n.saturating_add(written));
        if size.estimate.is_some_and(|n| n <= budget) {
            return;
        }
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return };
        let total = entries.flatten().filter_map(|entry| {
            let name = entry.file_name();
            let (key, ext) = name.to_str()?.split_once('.')?;
            if key.len() != 32 || !key.bytes().all(|b| b.is_ascii_hexdigit()) || !matches!(ext, "dxil" | "pipe") {
                return None;
            }
            let metadata = entry.metadata().ok()?;
            metadata.is_file().then_some(metadata.len())
        }).fold(0u64, u64::saturating_add);
        size.estimate = Some(total);
        if total > budget {
            size.warned = true;
            warn(total);
        }
    }
}

/// Hold the lock until process exit because the sampler registry loads its index map only once.
/// Remember a failed claim too, since that process may already be using an in-memory registry.
fn owns_sampler_registry(dir: &Path) -> bool {
    static OWNERS: Mutex<Option<HashMap<PathBuf, Option<File>>>> = Mutex::new(None);
    let mut owners = OWNERS.lock().unwrap_or_else(|e| e.into_inner());
    owners.get_or_insert_with(HashMap::new).entry(dir.to_owned()).or_insert_with(|| {
        std::fs::create_dir_all(dir).ok()?;
        let lock = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(dir.join("immediate-samplers.lock")).ok()?;
        lock.try_lock().ok()?;
        Some(lock)
    }).is_some()
}

fn non_empty(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn default_dir_from(
    windows: bool,
    local_appdata: Option<OsString>,
    user_profile: Option<OsString>,
    xdg_cache: Option<OsString>,
    home: Option<OsString>,
) -> Option<PathBuf> {
    let base = if windows {
        non_empty(local_appdata).or_else(|| non_empty(user_profile).map(|p| p.join("AppData").join("Local")))
    } else {
        non_empty(xdg_cache).or_else(|| non_empty(home).map(|h| h.join(".cache")))
    };
    base.map(|b| b.join("Decant").join("direct-dxil"))
}

/// `%LOCALAPPDATA%\Decant\direct-dxil` beside the run data on Windows, the user cache folder elsewhere.
pub fn default_dir() -> Option<PathBuf> {
    default_dir_from(
        cfg!(windows),
        std::env::var_os("LOCALAPPDATA"),
        std::env::var_os("USERPROFILE"),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::rts0::{RootParam, RootRange, StaticSampler};
    use crate::shader_source::{InputElement, StageInput};

    /// A scratch folder for one test, removed on drop.
    pub(crate) struct TempDir(pub PathBuf);

    impl TempDir {
        pub(crate) fn new(tag: &str) -> TempDir {
            static N: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!("d3d12xs-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&dir);
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample_pipeline(stages: &[(StageKind, &[u8])], root: RootLayout, targets: Vec<u32>) -> Pipeline {
        Pipeline {
            stages: stages.iter().map(|(k, b)| StageInput { kind: *k, bytes: Arc::from(*b) }).collect(),
            root,
            linkage: Linkage { stages: stages.iter().map(|(k, _)| *k).collect(), targets, ..Linkage::default() },
        }
    }

    #[test]
    fn fnv128_matches_the_published_vectors() {
        // FNV 1a 128 of the empty string is the offset basis, of "a" the published value.
        assert_eq!(Fnv128::default().finish(), FNV128_OFFSET);
        assert_eq!(Fnv128::default().bytes(b"a").finish(), 0xd228_cb69_6f1a_8caf_7891_2b70_4e4a_8964);
    }

    #[test]
    fn the_stage_and_pipeline_keys_are_stable_and_change_with_every_part() {
        let link = Linkage {
            stages: vec![StageKind::Vertex, StageKind::Pixel],
            inputs: vec![InputElement { semantic: "POSITION".into(), index: 0, format: 6, slot: 0, offset: 0, per_instance: false, step_rate: 0 }],
            targets: vec![28],
            depth_format: 40,
            topology_type: 3,
            sample_count: 1,
            stream_output: false,
        };
        let rich_root = RootLayout {
            version: 2,
            flags: 0x400,
            params: vec![
                RootParam { kind: RootParamKind::Table(vec![RootRange { class: RegisterClass::Srv, count: 1, base: 0, space: 0, flags: 0, offset: 0 }]), visibility: 0 },
                RootParam { kind: RootParamKind::Constants { register: 1, space: 0, count: 4 }, visibility: 1 },
                RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 },
            ],
            static_samplers: vec![StaticSampler { register: 0, space: 0, visibility: 5 }],
        };
        let key = |kind, bytes: &[u8], r: &RootLayout, l: &Linkage, t: &str, d: &str| stage_key(kind, bytes, r, l, t, d);
        let base = key(StageKind::Vertex, b"console bytes", &rich_root, &link, "t1", "d1");
        assert_eq!(base, key(StageKind::Vertex, b"console bytes", &rich_root, &link, "t1", "d1"));
        assert_eq!(format!("{base:032x}").len(), 32);

        let mut other_root = rich_root.clone();
        other_root.flags = 0;
        let mut other_sampler = rich_root.clone();
        other_sampler.static_samplers[0].register = 2;
        let other_link = Linkage { targets: vec![10], ..link.clone() };
        let other_input = Linkage {
            inputs: vec![InputElement { semantic: "TEXCOORD".into(), ..link.inputs[0].clone() }],
            ..link.clone()
        };
        for other in [
            key(StageKind::Pixel, b"console bytes", &rich_root, &link, "t1", "d1"),
            key(StageKind::Vertex, b"console bytez", &rich_root, &link, "t1", "d1"),
            key(StageKind::Vertex, b"console bytes", &other_root, &link, "t1", "d1"),
            key(StageKind::Vertex, b"console bytes", &other_sampler, &link, "t1", "d1"),
            key(StageKind::Vertex, b"console bytes", &rich_root, &other_link, "t1", "d1"),
            key(StageKind::Vertex, b"console bytes", &rich_root, &other_input, "t1", "d1"),
            key(StageKind::Vertex, b"console bytes", &rich_root, &link, "t2", "d1"),
            key(StageKind::Vertex, b"console bytes", &rich_root, &link, "t1", "d2"),
        ] {
            assert_ne!(other, base);
        }

        let p_base = sample_pipeline(&[(StageKind::Vertex, b"vs"), (StageKind::Pixel, b"ps")], rich_root.clone(), vec![28]);
        let pk = pipeline_key(&p_base, "t1", "d1");
        assert_eq!(pk, pipeline_key(&p_base, "t1", "d1"));
        for other in [
            pipeline_key(&sample_pipeline(&[(StageKind::Pixel, b"ps"), (StageKind::Vertex, b"vs")], rich_root.clone(), vec![28]), "t1", "d1"),
            pipeline_key(&sample_pipeline(&[(StageKind::Vertex, b"vs"), (StageKind::Pixel, b"pz")], rich_root.clone(), vec![28]), "t1", "d1"),
            pipeline_key(&sample_pipeline(&[(StageKind::Vertex, b"vs"), (StageKind::Pixel, b"ps")], other_root, vec![28]), "t1", "d1"),
            pipeline_key(&sample_pipeline(&[(StageKind::Vertex, b"vs"), (StageKind::Pixel, b"ps")], rich_root.clone(), vec![10]), "t1", "d1"),
            pipeline_key(&p_base, "t2", "d1"),
            pipeline_key(&p_base, "t1", "d2"),
        ] {
            assert_ne!(other, pk);
        }

        let by_source = source_stage_key(StageKind::Vertex, 7, "t1", "d1");
        assert_eq!(by_source, source_stage_key(StageKind::Vertex, 7, "t1", "d1"));
        for other in [
            source_stage_key(StageKind::Pixel, 7, "t1", "d1"),
            source_stage_key(StageKind::Vertex, 8, "t1", "d1"),
            source_stage_key(StageKind::Vertex, 7, "t2", "d1"),
            source_stage_key(StageKind::Vertex, 7, "t1", "d2"),
        ] {
            assert_ne!(other, by_source);
        }
    }

    #[test]
    fn a_written_entry_reads_back_and_a_damaged_or_old_one_is_a_miss() {
        let tmp = TempDir::new("cache");
        let cache = DiskCache::new(&tmp.0);
        assert_eq!(cache.read(7), None);
        assert!(cache.write(7, b"").is_err());
        cache.write(7, b"DXBC compiled").expect("write");
        assert_eq!(cache.read(7).as_deref(), Some(&b"DXBC compiled"[..]));
        assert_eq!(cache.read(8), None);

        // A file under another key's name is a miss.
        std::fs::copy(cache.path(7), cache.path(9)).expect("copy");
        assert_eq!(cache.read(9), None);

        // A flipped payload byte is caught by the payload checksum.
        let mut flipped = std::fs::read(cache.path(7)).expect("read");
        let end = flipped.len() - 1;
        flipped[end] ^= 0x01;
        std::fs::write(cache.path(7), &flipped).expect("write");
        assert_eq!(cache.read(7), None);

        // Writing again replaces the damaged file cleanly.
        cache.write(7, b"DXBC rebuilt").expect("rewrite");
        assert_eq!(cache.read(7).as_deref(), Some(&b"DXBC rebuilt"[..]));

        // An old format version in the header is a miss.
        let mut old = std::fs::read(cache.path(7)).expect("read");
        old[4..8].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(cache.path(7), &old).expect("write");
        assert_eq!(cache.read(7), None);

        // A cut file and a file with bad magic are misses.
        old.truncate(old.len() - 1);
        std::fs::write(cache.path(7), &old).expect("write");
        assert_eq!(cache.read(7), None);
        std::fs::write(cache.path(7), b"nonsense").expect("write");
        assert_eq!(cache.read(7), None);

        // Pipeline records round trip, survive sampler appends, and miss when samplers are reset.
        let samplers = tmp.0.join(SAMPLERS_FILE);
        std::fs::write(&samplers, b"SXSM\x01\x00\x00\x000123456789abcdef").expect("samplers");
        cache.write_pipeline(42, &[101, 202]).expect("write pipeline");
        assert_eq!(cache.read_pipeline(42), Some(vec![101, 202]));
        std::fs::write(&samplers, b"SXSM\x01\x00\x00\x000123456789abcdefMORE_SAMPLER_16B").expect("append sampler");
        assert_eq!(cache.read_pipeline(42), Some(vec![101, 202]), "appending new samplers keeps earlier indices valid");
        std::fs::write(&samplers, b"SXSM\x01\x00\x00\x00DIFFERENT_FIRST!").expect("replace sampler");
        assert_eq!(cache.read_pipeline(42), None, "reordered sampler registry invalidates the pipeline record");

        // No temporary file is left behind.
        let left: Vec<_> = std::fs::read_dir(&tmp.0).expect("dir").filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().ends_with(".tmp")).collect();
        assert!(left.is_empty());
    }

    #[test]
    fn a_failed_rename_leaves_an_entry_published_after_validation_intact() {
        let tmp = TempDir::new("rename_retry");
        let cache = DiskCache::new(&tmp.0);
        let first_validation = std::cell::Cell::new(true);
        let (checked, wait_for_check) = std::sync::mpsc::channel();
        let (installed, wait_for_install) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let cache = &cache;
            let writer = scope.spawn(move || {
                wait_for_check.recv().unwrap();
                cache.write(7, b"winning DXIL").unwrap();
                installed.send(()).unwrap();
            });
            let published = DiskCache::publish(
                || Err(io::Error::from(io::ErrorKind::PermissionDenied)),
                || {
                    let valid = cache.read(7).is_some();
                    if first_validation.replace(false) {
                        assert!(!valid);
                        checked.send(()).unwrap();
                        wait_for_install.recv().unwrap();
                    }
                    valid
                },
            ).unwrap();
            assert!(!published);
            assert_eq!(cache.read(7).as_deref(), Some(&b"winning DXIL"[..]));
            writer.join().unwrap();
        });
    }

    #[test]
    fn an_oversized_stage_with_a_valid_header_is_a_miss() {
        let tmp = TempDir::new("oversized");
        let cache = DiskCache::new(&tmp.0);
        cache.write(7, b"DXBC").unwrap();
        let path = cache.path(7);
        let mut file = OpenOptions::new().write(true).open(path).unwrap();
        let length = MAX_DXIL_BYTES as u64 * 16;
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(24)).unwrap();
        file.write_all(&length.to_le_bytes()).unwrap();
        file.set_len(STAGE_HEADER as u64 + length).unwrap();
        assert_eq!(cache.read(7), None);
    }

    #[test]
    fn the_size_warning_counts_old_formats_and_fires_once() {
        let tmp = TempDir::new("size");
        let cache = DiskCache::new(&tmp.0);
        let monitor = DiskCache::new(&tmp.0);
        let budget = 100;
        let mut warnings = Vec::new();
        cache.write(1, b"DXBC").unwrap();
        let mut old = std::fs::read(cache.path(1)).unwrap();
        old[4..8].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(cache.path(1), &old).unwrap();
        monitor.check_size(0, budget, |n| warnings.push(n));
        assert!(warnings.is_empty());

        cache.write_pipeline(2, &[1]).unwrap();
        monitor.check_size((PIPE_HEADER + 16) as u64, budget, |n| warnings.push(n));
        assert_eq!(warnings, [124]);
        cache.write_pipeline(3, &[1]).unwrap();
        monitor.check_size((PIPE_HEADER + 16) as u64, budget, |n| warnings.push(n));
        assert_eq!(warnings, [124]);
    }

    #[test]
    fn the_folder_comes_from_the_switch_or_the_default() {
        let base = std::env::temp_dir().join("dflt");
        let default = Some(base.clone());
        assert_eq!(DiskCache::resolve_dir(None, default.clone()), default);
        assert_eq!(DiskCache::resolve_dir(Some("".into()), default.clone()), default);
        assert_eq!(DiskCache::resolve_dir(Some("mine".into()), default.clone()), Some(base.join("mine")));
        assert!(DiskCache::resolve_dir(Some("0".into()), default).is_none());
        assert!(DiskCache::resolve_dir(None, None).is_none());

        assert_eq!(
            default_dir_from(true, Some("local_root".into()), Some("profile_root".into()), None, None),
            Some(PathBuf::from("local_root").join("Decant").join("direct-dxil"))
        );
        assert_eq!(
            default_dir_from(true, Some("".into()), Some("profile_root".into()), None, None),
            Some(PathBuf::from("profile_root").join("AppData").join("Local").join("Decant").join("direct-dxil"))
        );
        assert_eq!(
            default_dir_from(false, None, None, Some("".into()), Some("home_root".into())),
            Some(PathBuf::from("home_root").join(".cache").join("Decant").join("direct-dxil"))
        );
    }
}
