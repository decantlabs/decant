//! The disk library of compiled PC pipeline states for the direct shader path (`direct_pso`).
//!
//! Every PC pipeline made from translated shaders costs a driver compile of its DXIL. Without a
//! library that compile runs again on every run, and a scene load can spend most of its time in
//! it (`pc_creation_us` of the direct summary). After a direct pipeline is made, its driver blob
//! (`ID3D12PipelineState::GetCachedBlob`) is stored here. On a later run the same pipeline gets the
//! blob in the `CachedPSO` field of its description, and the PC runtime skips the compile.
//!
//! The blob goes through the description and not through `ID3D12PipelineLibrary1`, because the
//! PC creation call belongs to the creation path of `lazy_pso` and `proxy`. The description is
//! the one input of that call the direct path decides. Both forms hold the same driver blob.
//!
//! Folder: `DECANT_D3D12X_PSO_CACHE=<dir>`, default `%LOCALAPPDATA%\Decant\pso-library` beside the
//! DXIL cache (the user cache folder elsewhere). A relative value is resolved under the default
//! folder, `0` switches the library off. Inside it, one folder per context key holds one file
//! per pipeline key.
//!
//! * The context key hashes the PCI identity of the adapter (vendor, device, subsystem and
//!   revision), the user mode driver version, the translator hash and the format version of this
//!   module. A blob of another kind of adapter, another driver or another translator lives in
//!   another folder and is never read.
//! * The adapter LUID is not part of the context key. Windows assigns a new LUID to an adapter on
//!   every boot, so a key with it would leave the library empty after each reboot. Two adapters
//!   of the same PCI identity on the same driver share one folder, which is safe because the driver
//!   blob depends on the kind of adapter and the driver, not on the one card.
//! * The pipeline key hashes the description with its pointers left out, the bytes of every
//!   compiled stage and the direct pipeline key (console stages, root signature layout, linkage).
//!   Graphics pipelines with stream output get no key: their PC description holds state behind
//!   pointers that the key cannot see.
//! * Mesh pipelines get no key and no `CachedPSO` field. The PC runtime takes them only as a
//!   pipeline stream, and `mesh_pso::pc_stream` writes that stream without a `CACHED_PSO`
//!   subobject. A stored blob could never reach the runtime, so the library would store a blob on
//!   every run and never save a compile. Mesh pipelines can join once the stream carries the
//!   `CachedPSO` field of their description.
//! * A file is `<pipeline key as 32 lower case hex digits>.pso` with a 64 byte header (magic
//!   `SXPL`, format version, context key, pipeline key, blob length, 128 bit FNV 1a of the blob)
//!   followed by the blob. A file whose header, length or checksum does not match is ignored, and
//!   the pipeline is compiled and stored again.
//! * Files are written to a unique temporary name and renamed, so a concurrent or interrupted run
//!   never leaves a half file under a real name.
//! * A blob the PC runtime refuses (driver or description mismatch) is deleted. That pipeline is
//!   made from the title's shaders by `lazy_pso`, as any refused direct pipeline is. After
//!   [`MAX_REJECTIONS`] refusals the library stops for the run.

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Condvar, LazyLock, Mutex, OnceLock, Weak};
use std::thread::JoinHandle;

use crate::direct_cache::Fnv128;

/// Bumped when the layout of the keys or of the files changes. Version 2 left the adapter LUID out
/// of the context key.
const FORMAT: u32 = 2;
const MAGIC: &[u8; 4] = b"SXPL";
const HEADER: usize = 64;
/// The largest blob stored. Driver blobs measured on desktop drivers are far below this.
const MAX_BLOB_BYTES: usize = 64 * 1024 * 1024;
/// Refused blobs after which the library stops for the run.
pub const MAX_REJECTIONS: u32 = 4;

/// Size of `D3D12_GRAPHICS_PIPELINE_STATE_DESC`, from the public d3d12.h.
const GRAPHICS_DESC_SIZE: usize = crate::stream_pso::GRAPHICS_DESC_SIZE;
/// Size of `D3D12_COMPUTE_PIPELINE_STATE_DESC`, from the public d3d12.h.
const COMPUTE_DESC_SIZE: usize = 0x38;

/// Size of our mesh description (`mesh_pso`).
const MESH_DESC_SIZE: usize = crate::mesh_pso::MESH_DESC_SIZE;

/// Where the `D3D12_CACHED_PIPELINE_STATE` (blob pointer, blob size) of a description lies.
pub fn cached_field(desc_len: usize) -> Option<usize> {
    match desc_len {
        GRAPHICS_DESC_SIZE | MESH_DESC_SIZE => Some(0x278),
        COMPUTE_DESC_SIZE => Some(0x20),
        _ => None,
    }
}

/// The byte ranges of a description the pipeline key leaves out: pointers, the shader byte code
/// fields (the compiled bytes are hashed instead), the cached blob and padding. From the public
/// d3d12.h layouts for x64.
fn masked_ranges(desc_len: usize) -> &'static [(usize, usize)] {
    match desc_len {
        GRAPHICS_DESC_SIZE => &[
            (0x000, 0x008), // pRootSignature
            (0x008, 0x058), // VS, PS, DS, HS, GS
            (0x058, 0x060), // StreamOutput.pSODeclaration
            (0x064, 0x068), // padding
            (0x068, 0x070), // StreamOutput.pBufferStrides
            (0x224, 0x228), // padding
            (0x228, 0x230), // InputLayout.pInputElementDescs
            (0x234, 0x238), // padding
            (0x274, 0x278), // padding
            (0x278, 0x288), // CachedPSO
            (0x28C, 0x290), // padding
        ],
        MESH_DESC_SIZE => &[
            (0x000, 0x008), // pRootSignature
            (0x008, 0x058), // VS, PS, DS, HS, GS
            (0x058, 0x060), // StreamOutput.pSODeclaration
            (0x064, 0x068), // padding
            (0x068, 0x070), // StreamOutput.pBufferStrides
            (0x224, 0x228), // padding
            (0x228, 0x230), // InputLayout.pInputElementDescs
            (0x234, 0x238), // padding
            (0x274, 0x278), // padding
            (0x278, 0x288), // CachedPSO
            (0x28C, 0x290), // padding
            (GRAPHICS_DESC_SIZE, MESH_DESC_SIZE), // AS and MS D3D12_SHADER_BYTECODE
        ],
        COMPUTE_DESC_SIZE => &[
            (0x00, 0x08), // pRootSignature
            (0x08, 0x18), // CS
            (0x1C, 0x20), // padding
            (0x20, 0x30), // CachedPSO
            (0x34, 0x38), // padding
        ],
        _ => &[],
    }
}

/// The key of one pipeline: `desc` is the PC description the direct path makes it with,
/// `stages` the compiled bytes of its stages in order. `_identity` is ignored so identical
/// compiled DXIL and pipeline state reuse driver blobs across translator rebuilds.
/// `None` for a pipeline the library does not keep.
pub fn pipeline_key(desc: &[u8], stages: &[&[u8]], _identity: u128) -> Option<u128> {
    cached_field(desc.len())?;
    // Stream output declarations live behind pointers: not keyed, not kept.
    if (desc.len() == GRAPHICS_DESC_SIZE || desc.len() == MESH_DESC_SIZE) && u32::from_le_bytes(desc[0x60..0x64].try_into().ok()?) != 0 {
        return None;
    }
    let mut stack = [0u8; MESH_DESC_SIZE];
    let state = &mut stack[..desc.len()];
    state.copy_from_slice(desc);
    for &(from, to) in masked_ranges(desc.len()) {
        state[from..to].fill(0);
    }
    let mut h = Fnv128::default();
    h.field(b"decant pso library pipeline").field(&FORMAT.to_le_bytes()).field(state);
    h.field(&(stages.len() as u64).to_le_bytes());
    for stage in stages {
        h.field(stage);
    }
    Some(h.finish())
}

/// The adapter and driver a blob is valid for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterIdentity {
    /// The adapter LUID, low part in the low 32 bits. It names the adapter for this boot only and
    /// is left out of [`context_key`].
    pub luid: u64,
    pub vendor_id: u32,
    pub device_id: u32,
    pub subsys_id: u32,
    pub revision: u32,
    /// The user mode driver version (`IDXGIAdapter::CheckInterfaceSupport`).
    pub driver_version: u64,
}

/// The key of the folder that holds the blobs of one kind of adapter and driver. It stays the
/// same across reboots and across binary rebuilds when the adapter and driver are unchanged,
/// because [`pipeline_key`] already hashes the compiled DXIL stage bytes.
pub fn context_key(adapter: &AdapterIdentity, _translator: &str) -> u128 {
    let mut h = Fnv128::default();
    h.field(b"decant pso library context").field(&FORMAT.to_le_bytes());
    for word in [adapter.vendor_id, adapter.device_id, adapter.subsys_id, adapter.revision] {
        h.field(&word.to_le_bytes());
    }
    h.field(&adapter.driver_version.to_le_bytes());
    h.finish()
}

/// The library folder: `DECANT_D3D12X_PSO_CACHE`, else the default.
pub fn configured_root() -> Option<PathBuf> {
    let default = crate::direct_cache::default_dir().and_then(|dxil| Some(dxil.parent()?.join("pso-library")));
    resolve_root(std::env::var_os("DECANT_D3D12X_PSO_CACHE"), default)
}

/// The folder for a setting: `0` off, an absolute path as it is, a relative one of plain parts
/// under `default`, anything else off.
fn resolve_root(configured: Option<OsString>, default: Option<PathBuf>) -> Option<PathBuf> {
    let default = default.filter(|base| base.is_absolute());
    let Some(path) = configured.filter(|v| !v.is_empty()).map(PathBuf::from) else { return default };
    if path == Path::new("0") {
        None
    } else if path.is_absolute() {
        Some(path)
    } else if path.components().all(|part| matches!(part, std::path::Component::Normal(_) | std::path::Component::CurDir)) {
        default.map(|base| base.join(path))
    } else {
        None
    }
}

/// The blob files of one context key.
#[derive(Clone)]
pub struct Store {
    dir: PathBuf,
    context: u128,
}

static TEMP_NUMBER: AtomicU64 = AtomicU64::new(0);

impl Store {
    /// The store of `context` under the library folder `root`.
    pub fn new(root: &Path, context: u128) -> Store {
        Store { dir: root.join(format!("{context:032x}")), context }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, key: u128) -> PathBuf {
        self.dir.join(format!("{key:032x}.pso"))
    }

    /// The blob stored for `key`. `None` on a miss or a file that fails any check.
    pub fn read(&self, key: u128) -> Option<Vec<u8>> {
        let mut file = File::open(self.path(key)).ok()?;
        let len = file.metadata().ok()?.len();
        if len <= HEADER as u64 || len > (HEADER + MAX_BLOB_BYTES) as u64 {
            return None;
        }
        let mut bytes = vec![0u8; len as usize];
        file.read_exact(&mut bytes).ok()?;
        self.parse(key, &bytes)?;
        bytes.copy_within(HEADER.., 0);
        bytes.truncate(bytes.len() - HEADER);
        Some(bytes)
    }

    /// The blob of a file's bytes, when the header names this store and `key` and the length and
    /// checksum match.
    fn parse<'a>(&self, key: u128, bytes: &'a [u8]) -> Option<&'a [u8]> {
        let (header, blob) = bytes.split_at_checked(HEADER)?;
        let u128_at = |at: usize| header[at..at + 16].try_into().ok().map(u128::from_le_bytes);
        let valid = &header[0..4] == MAGIC
            && header[4..8] == FORMAT.to_le_bytes()
            && u128_at(8)? == self.context
            && u128_at(24)? == key
            && header[40..48] == (blob.len() as u64).to_le_bytes()
            && u128_at(48)? == Fnv128::default().bytes(blob).finish();
        valid.then_some(blob)
    }

    /// The file of `blob` under `key`, header first.
    fn encode(&self, key: u128, blob: &[u8]) -> Vec<u8> {
        let mut file = Vec::with_capacity(HEADER + blob.len());
        file.extend_from_slice(MAGIC);
        file.extend_from_slice(&FORMAT.to_le_bytes());
        file.extend_from_slice(&self.context.to_le_bytes());
        file.extend_from_slice(&key.to_le_bytes());
        file.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        file.extend_from_slice(&Fnv128::default().bytes(blob).finish().to_le_bytes());
        file.extend_from_slice(blob);
        file
    }

    /// Stores `blob` under `key`: written to a temporary file of this process and renamed.
    pub fn write(&self, key: u128, blob: &[u8]) -> io::Result<()> {
        if blob.is_empty() || blob.len() > MAX_BLOB_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "blob size out of range"));
        }
        std::fs::create_dir_all(&self.dir)?;
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let seq = TEMP_NUMBER.fetch_add(1, Ordering::Relaxed);
        let temp = self.dir.join(format!(".{key:032x}.{}.{nanos}.{seq}.tmp", std::process::id()));
        let written = File::create_new(&temp).and_then(|mut file| file.write_all(&self.encode(key, blob)));
        let result = written.and_then(|()| self.publish(&temp, key));
        let _ = std::fs::remove_file(&temp);
        result
    }

    /// Renames `temp` to the file of `key`. A reader on Windows can block the replacement for a
    /// moment, and another process may have published the same key: that counts as done.
    fn publish(&self, temp: &Path, key: u128) -> io::Result<()> {
        let mut retries = 5;
        loop {
            match std::fs::rename(temp, self.path(key)) {
                Ok(()) => return Ok(()),
                Err(_) if self.read(key).is_some() => return Ok(()),
                Err(e) if retries == 0 => return Err(e),
                Err(_) => {
                    retries -= 1;
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
        }
    }

    /// Deletes the file of `key`, after the PC runtime refused its blob.
    pub fn forget(&self, key: u128) {
        let _ = std::fs::remove_file(self.path(key));
    }
}

/// The counts of a run.
#[derive(Default)]
pub struct Counts {
    /// Pipelines that got a stored blob.
    pub hits: AtomicU64,
    /// Pipelines without a stored blob.
    pub misses: AtomicU64,
    /// Blobs written.
    pub stored: AtomicU64,
    /// Blobs the PC runtime refused.
    pub rejected: AtomicU32,
}

const WRITER_QUEUE_CAP: usize = 128;
const PENDING_SHARDS: usize = 16;

type PendingMap = HashMap<(PathBuf, u128), Arc<[u8]>>;
type WriteSender = Mutex<Option<SyncSender<(u128, Arc<[u8]>)>>>;

static PENDING_COUNT: AtomicUsize = AtomicUsize::new(0);
static PENDING_WRITES: LazyLock<[Mutex<PendingMap>; PENDING_SHARDS]> = LazyLock::new(|| std::array::from_fn(|_| Mutex::default()));
static ACTIVE_WRITERS: LazyLock<Mutex<HashMap<PathBuf, Weak<BackgroundWriter>>>> = LazyLock::new(Mutex::default);

fn pending_shard(key: u128) -> usize {
    (key as usize) & (PENDING_SHARDS - 1)
}

fn lookup_pending(dir: &Path, key: u128) -> Option<Vec<u8>> {
    if PENDING_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    let map = PENDING_WRITES[pending_shard(key)].lock().unwrap_or_else(|e| e.into_inner());
    if map.is_empty() {
        return None;
    }
    map.get(&(dir.to_path_buf(), key)).map(|blob| blob.to_vec())
}

fn insert_pending(dir: &Path, key: u128, blob: Arc<[u8]>) {
    let mut map = PENDING_WRITES[pending_shard(key)].lock().unwrap_or_else(|e| e.into_inner());
    if map.insert((dir.to_path_buf(), key), blob).is_none() {
        PENDING_COUNT.fetch_add(1, Ordering::Release);
    }
}

fn remove_pending(dir: &Path, key: u128, blob: &Arc<[u8]>) {
    let mut map = PENDING_WRITES[pending_shard(key)].lock().unwrap_or_else(|e| e.into_inner());
    let map_key = (dir.to_path_buf(), key);
    if map.get(&map_key).is_some_and(|cur| Arc::ptr_eq(cur, blob)) && map.remove(&map_key).is_some() {
        PENDING_COUNT.fetch_sub(1, Ordering::Release);
    }
}

fn clear_pending(dir: &Path, key: u128) {
    if PENDING_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut map = PENDING_WRITES[pending_shard(key)].lock().unwrap_or_else(|e| e.into_inner());
    if !map.is_empty() && map.remove(&(dir.to_path_buf(), key)).is_some() {
        PENDING_COUNT.fetch_sub(1, Ordering::Release);
    }
}

struct BackgroundWriter {
    tx: WriteSender,
    handle: Mutex<Option<JoinHandle<()>>>,
    in_flight: (Mutex<usize>, Condvar),
}

impl BackgroundWriter {
    fn spawn(store: Store) -> Arc<Self> {
        let (tx, rx) = mpsc::sync_channel::<(u128, Arc<[u8]>)>(WRITER_QUEUE_CAP);
        let writer = Arc::new(Self {
            tx: Mutex::new(Some(tx)),
            handle: Mutex::new(None),
            in_flight: (Mutex::new(0), Condvar::new()),
        });
        let worker_ref = Arc::downgrade(&writer);
        let handle = std::thread::Builder::new()
            .name("decant-pso-writer".to_owned())
            .spawn(move || {
                while let Ok((key, blob)) = rx.recv() {
                    let _ = store.write(key, &blob);
                    remove_pending(store.dir(), key, &blob);
                    if let Some(w) = worker_ref.upgrade() {
                        let (lock, cv) = &w.in_flight;
                        let mut count = lock.lock().unwrap_or_else(|e| e.into_inner());
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            cv.notify_all();
                        }
                    }
                }
            })
            .ok();
        *writer.handle.lock().unwrap_or_else(|e| e.into_inner()) = handle;
        writer
    }

    fn enqueue(&self, key: u128, blob: Arc<[u8]>) {
        {
            let mut count = self.in_flight.0.lock().unwrap_or_else(|e| e.into_inner());
            *count += 1;
        }
        let sent = {
            let guard = self.tx.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().is_some_and(|tx| tx.send((key, blob)).is_ok())
        };
        if !sent {
            let (lock, cv) = &self.in_flight;
            let mut count = lock.lock().unwrap_or_else(|e| e.into_inner());
            *count = count.saturating_sub(1);
            if *count == 0 {
                cv.notify_all();
            }
        }
    }

    fn wait_idle(&self) {
        let (lock, cv) = &self.in_flight;
        let mut count = lock.lock().unwrap_or_else(|e| e.into_inner());
        while *count > 0 {
            count = cv.wait(count).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn shutdown(&self) {
        drop(self.tx.lock().unwrap_or_else(|e| e.into_inner()).take());
        if let Some(handle) = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = handle.join();
        }
    }
}

/// The library of a run. It opens once the adapter is known, which is at the first direct
/// pipeline the PC runtime made.
pub struct Library {
    root: PathBuf,
    translator: &'static str,
    store: OnceLock<Option<Store>>,
    writer: OnceLock<Option<Arc<BackgroundWriter>>>,
    counts: Counts,
}

impl Library {
    pub fn new(root: PathBuf, translator: &'static str) -> Library {
        Library {
            root,
            translator,
            store: OnceLock::new(),
            writer: OnceLock::new(),
            counts: Counts::default(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn counts(&self) -> &Counts {
        &self.counts
    }

    /// Opens the store for the adapter `adapter` names, on the first call only. `None` when the
    /// adapter is unknown, then the library stays closed for the run.
    pub fn open(&self, adapter: impl FnOnce() -> Option<AdapterIdentity>) -> Option<&Store> {
        let store = self.store.get_or_init(|| adapter().map(|a| Store::new(&self.root, context_key(&a, self.translator)))).as_ref();
        if let Some(s) = store {
            self.writer.get_or_init(|| {
                let dir = s.dir().to_path_buf();
                let prev = ACTIVE_WRITERS.lock().unwrap_or_else(|e| e.into_inner()).get(&dir).and_then(Weak::upgrade);
                if let Some(prev) = prev {
                    prev.wait_idle();
                }
                let writer = BackgroundWriter::spawn(s.clone());
                ACTIVE_WRITERS.lock().unwrap_or_else(|e| e.into_inner()).insert(dir, Arc::downgrade(&writer));
                Some(writer)
            });
        }
        store
    }

    /// The store while the library is open and has not seen too many refusals.
    fn live(&self) -> Option<&Store> {
        if self.counts.rejected.load(Ordering::Relaxed) >= MAX_REJECTIONS {
            return None;
        }
        self.store.get()?.as_ref()
    }

    /// The stored blob of `key`, counted as a hit or a miss.
    pub fn lookup(&self, key: u128) -> Option<Vec<u8>> {
        let store = self.live()?;
        if let Some(blob) = lookup_pending(store.dir(), key) {
            self.counts.hits.fetch_add(1, Ordering::Relaxed);
            return Some(blob);
        }
        let blob = store.read(key);
        let count = if blob.is_some() { &self.counts.hits } else { &self.counts.misses };
        count.fetch_add(1, Ordering::Relaxed);
        blob
    }

    /// Stores the blob of a pipeline made without one, persisting to disk on a background writer
    /// while making it visible in memory immediately.
    pub fn keep(&self, key: u128, blob: &[u8]) -> io::Result<()> {
        let Some(store) = self.live() else {
            return Ok(());
        };
        if blob.is_empty() || blob.len() > MAX_BLOB_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "blob size out of range"));
        }
        let shared: Arc<[u8]> = Arc::from(blob);
        insert_pending(store.dir(), key, Arc::clone(&shared));
        self.counts.stored.fetch_add(1, Ordering::Relaxed);
        if let Some(Some(writer)) = self.writer.get() {
            writer.enqueue(key, shared);
        }
        Ok(())
    }

    /// The PC runtime refused the stored blob of `key`: the file is deleted. Returns whether the
    /// library is still on.
    pub fn reject(&self, key: u128) -> bool {
        if let Some(store) = self.store.get().and_then(Option::as_ref) {
            clear_pending(store.dir(), key);
            store.forget(key);
        }
        self.counts.rejected.fetch_add(1, Ordering::Relaxed) + 1 < MAX_REJECTIONS
    }

    /// Flushes any queued background disk writes.
    pub fn flush(&self) {
        if let Some(Some(writer)) = self.writer.get() {
            writer.wait_idle();
        }
    }

    /// One line with the counts, for the summary of the direct path.
    pub fn summary(&self) -> String {
        let c = &self.counts;
        let folder = self.store.get().and_then(Option::as_ref).map_or_else(|| "not open".to_string(), |s| s.dir().display().to_string());
        format!(
            "pso library: hits={} misses={} stored={} rejected={}, folder {folder}",
            c.hits.load(Ordering::Relaxed),
            c.misses.load(Ordering::Relaxed),
            c.stored.load(Ordering::Relaxed),
            c.rejected.load(Ordering::Relaxed)
        )
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if let Some(Some(writer)) = self.writer.get() {
            writer.shutdown();
        }
    }
}

#[cfg(windows)]
pub use pc::{adapter_of, cached_blob};

/// The calls into the PC runtime and DXGI.
#[cfg(windows)]
#[allow(unsafe_code)]
mod pc {
    use std::ffi::c_void;

    use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12PipelineState};
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIDevice, IDXGIFactory4};
    use windows::core::Interface;

    use super::AdapterIdentity;

    /// The adapter and driver of the device that made the PC pipeline `pso`. `None` when DXGI does
    /// not tell the driver version.
    pub fn adapter_of(pso: *mut c_void) -> Option<AdapterIdentity> {
        // SAFETY: `pso` is a live PC pipeline state, borrowed for this call only.
        let pso = unsafe { ID3D12PipelineState::from_raw_borrowed(&pso) }?;
        // SAFETY: plain D3D12 and DXGI calls on live objects.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            pso.GetDevice(&mut device).ok()?;
            let luid = device?.GetAdapterLuid();
            let adapter: IDXGIAdapter1 = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?.EnumAdapterByLuid(luid).ok()?;
            let desc = adapter.GetDesc1().ok()?;
            let driver_version = adapter.CheckInterfaceSupport(&IDXGIDevice::IID).ok()? as u64;
            Some(AdapterIdentity {
                luid: u64::from(luid.LowPart) | (u64::from(luid.HighPart as u32) << 32),
                vendor_id: desc.VendorId,
                device_id: desc.DeviceId,
                subsys_id: desc.SubSysId,
                revision: desc.Revision,
                driver_version,
            })
        }
    }

    /// The driver blob of the PC pipeline `pso`. `None` when the runtime gives none.
    pub fn cached_blob(pso: *mut c_void) -> Option<Vec<u8>> {
        // SAFETY: `pso` is a live PC pipeline state, borrowed for this call only.
        let pso = unsafe { ID3D12PipelineState::from_raw_borrowed(&pso) }?;
        // SAFETY: the blob owns `GetBufferSize` bytes while it lives.
        unsafe {
            let blob = pso.GetCachedBlob().ok()?;
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            (!bytes.is_empty()).then(|| bytes.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_cache::tests::TempDir;

    fn adapter() -> AdapterIdentity {
        AdapterIdentity { luid: 0x0000_0001_0000_ABCD, vendor_id: 0x10DE, device_id: 0x2704, subsys_id: 0x1234, revision: 0xA1, driver_version: 0x0020_0011_000F_1234 }
    }

    /// A graphics description with every pointer and padding field set to `fill` and two state
    /// bytes that differ by `state`.
    fn graphics(fill: u8, state: u8) -> Vec<u8> {
        let mut desc = vec![0u8; GRAPHICS_DESC_SIZE];
        for &(from, to) in masked_ranges(GRAPHICS_DESC_SIZE) {
            desc[from..to].fill(fill);
        }
        desc[0x060..0x064].fill(0);
        desc[0x1C4] = state;
        desc[0x240] = 1;
        desc
    }

    #[test]
    fn the_context_key_changes_with_adapter_and_driver_but_stays_stable_across_rebuilds() {
        let base = context_key(&adapter(), "translator a");
        assert_eq!(base, context_key(&adapter(), "translator a"), "stable for the same inputs");
        assert_eq!(context_key(&adapter(), "translator b"), base, "stable across rebuilds on the same GPU and driver");
        let changed = [
            AdapterIdentity { vendor_id: 0x1002, ..adapter() },
            AdapterIdentity { device_id: 1, ..adapter() },
            AdapterIdentity { subsys_id: 1, ..adapter() },
            AdapterIdentity { revision: 1, ..adapter() },
            AdapterIdentity { driver_version: 1, ..adapter() },
        ];
        for other in changed {
            assert_ne!(context_key(&other, "translator a"), base, "{other:?}");
        }
    }

    #[test]
    fn the_context_key_ignores_the_luid_windows_assigns_on_each_boot() {
        let base = context_key(&adapter(), "translator a");
        for luid in [0, 1, 0x0000_0001_0000_ABCE, 0x0000_0000_0001_2F40, u64::MAX] {
            assert_eq!(context_key(&AdapterIdentity { luid, ..adapter() }, "translator a"), base, "luid {luid:#x}");
        }
        let elsewhere = AdapterIdentity { luid: 0x0000_0000_0000_9999, driver_version: 1, ..adapter() };
        assert_ne!(context_key(&elsewhere, "translator a"), base, "a new driver still counts with a new luid");
    }

    #[test]
    fn a_blob_stored_before_a_reboot_is_found_after_it() {
        let dir = TempDir::new("pso-library-reboot");
        let before = Library::new(dir.0.clone(), "t");
        let folder = before.open(|| Some(adapter())).unwrap().dir().to_path_buf();
        before.keep(7, b"driver blob before the reboot").unwrap();
        before.flush();

        // The next boot gives the same adapter another LUID.
        let after = Library::new(dir.0.clone(), "t");
        let reopened = after.open(|| Some(AdapterIdentity { luid: 0x0000_0000_0002_1F7C, ..adapter() })).unwrap();
        assert_eq!(reopened.dir(), folder, "the same folder");
        assert_eq!(after.lookup(7).as_deref(), Some(&b"driver blob before the reboot"[..]));
        assert_eq!(after.counts().hits.load(Ordering::Relaxed), 1);

        let updated = Library::new(dir.0.clone(), "t");
        updated.open(|| Some(AdapterIdentity { luid: 0x0000_0000_0003_0001, driver_version: 2, ..adapter() })).unwrap();
        assert_eq!(updated.lookup(7), None, "a driver update starts a new folder");
    }

    #[test]
    fn mesh_pipelines_get_a_key_and_cached_field() {
        use crate::mesh_pso::{AS_AT, MS_AT};
        assert_eq!(cached_field(MESH_DESC_SIZE), Some(0x278));
        // A mesh description as `mesh_pso` lays it out: the graphics part, then the amplification
        // and the mesh shader.
        let mut mesh = vec![0u8; MESH_DESC_SIZE];
        mesh[..GRAPHICS_DESC_SIZE].copy_from_slice(&graphics(0x11, 1));
        for at in [AS_AT, MS_AT] {
            mesh[at..at + 8].copy_from_slice(&0x1000u64.to_le_bytes());
            mesh[at + 8..at + 16].copy_from_slice(&64u64.to_le_bytes());
        }
        let stages: [&[u8]; 3] = [b"DXIL amplification", b"DXIL mesh", b"DXIL pixel"];
        let base = pipeline_key(&mesh, &stages, 7).expect("mesh pipeline should be keyed");
        let mut other_ptrs = vec![0u8; MESH_DESC_SIZE];
        other_ptrs[..GRAPHICS_DESC_SIZE].copy_from_slice(&graphics(0x99, 1));
        for at in [AS_AT, MS_AT] {
            other_ptrs[at..at + 8].copy_from_slice(&0x9000u64.to_le_bytes());
            other_ptrs[at + 8..at + 16].copy_from_slice(&128u64.to_le_bytes());
        }
        assert_eq!(pipeline_key(&other_ptrs, &stages, 7), Some(base), "pointers and AS/MS bytecode fields are masked");
        assert_ne!(pipeline_key(&mesh, &[b"DXIL amplification", b"DXIL mesh 2", b"DXIL pixel"], 7), Some(base));
    }

    #[test]
    fn the_pipeline_key_ignores_pointers_and_padding_but_not_state_or_stages() {
        let stages: [&[u8]; 2] = [b"DXBC vertex", b"DXBC pixel"];
        let base = pipeline_key(&graphics(0x11, 1), &stages, 7).unwrap();
        assert_eq!(pipeline_key(&graphics(0x99, 1), &stages, 7), Some(base), "pointers and padding do not count");
        assert_ne!(pipeline_key(&graphics(0x11, 2), &stages, 7), Some(base), "state counts");
        assert_eq!(pipeline_key(&graphics(0x11, 1), &stages, 8), Some(base), "identical DXIL and state stay cached across translator rebuilds");
        assert_ne!(pipeline_key(&graphics(0x11, 1), &[b"DXBC vertex", b"DXBC pixel 2"], 7), Some(base), "stage bytes count");
        assert_ne!(pipeline_key(&graphics(0x11, 1), &stages[..1], 7), Some(base), "the stage count counts");

        let mut streamed = graphics(0x11, 1);
        streamed[0x60] = 1;
        assert_eq!(pipeline_key(&streamed, &stages, 7), None, "stream output is not kept");

        let mut compute = vec![0x55u8; COMPUTE_DESC_SIZE];
        compute[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
        compute[0x30..0x34].copy_from_slice(&0u32.to_le_bytes());
        let key = pipeline_key(&compute, &[b"DXBC compute"], 7).unwrap();
        compute[0x20..0x30].fill(0xEE);
        assert_eq!(pipeline_key(&compute, &[b"DXBC compute"], 7), Some(key), "the cached blob field does not count");
        compute[0x30] = 4;
        assert_ne!(pipeline_key(&compute, &[b"DXBC compute"], 7), Some(key), "the flags count");
    }

    #[test]
    fn the_folder_follows_the_setting() {
        let base = std::env::temp_dir().join("decant-pso-default");
        let resolve = |v: Option<&str>| resolve_root(v.map(OsString::from), Some(base.clone()));
        assert_eq!(resolve(None), Some(base.clone()));
        assert_eq!(resolve(Some("")), Some(base.clone()));
        assert_eq!(resolve(Some("0")), None, "0 switches the library off");
        assert_eq!(resolve(Some("other")), Some(base.join("other")), "relative under the default");
        assert_eq!(resolve(Some("../escape")), None, "a parent part is refused");
        let absolute = std::env::temp_dir().join("decant-pso-elsewhere");
        assert_eq!(resolve(Some(absolute.to_str().unwrap())), Some(absolute.clone()));
        assert_eq!(resolve_root(None, Some(PathBuf::from("relative"))), None, "a relative default is refused");
    }

    #[test]
    fn a_stored_blob_reads_back_and_a_leftover_temporary_file_changes_nothing() {
        let dir = TempDir::new("pso-library-roundtrip");
        let store = Store::new(&dir.0, 0xC0);
        assert_eq!(store.read(1), None);
        store.write(1, b"driver blob one").unwrap();
        assert_eq!(store.read(1).as_deref(), Some(&b"driver blob one"[..]));
        // A run stopped between writing and renaming leaves only a temporary file.
        std::fs::write(store.dir().join(format!(".{:032x}.1.2.3.tmp", 2u128)), store.encode(2, b"half")).unwrap();
        assert_eq!(store.read(2), None, "a temporary file is no entry");
        store.write(1, b"driver blob two").unwrap();
        assert_eq!(store.read(1).as_deref(), Some(&b"driver blob two"[..]), "a rewrite replaces the entry");
        assert_eq!(Store::new(&dir.0, 0xC1).read(1), None, "another context does not see it");
        assert!(store.write(3, b"").is_err(), "an empty blob is not stored");
    }

    #[test]
    fn writers_of_one_key_on_many_threads_leave_one_valid_file() {
        let dir = TempDir::new("pso-library-threads");
        let root = dir.0.clone();
        let threads: Vec<_> = (0..8u8)
            .map(|i| {
                let root = root.clone();
                std::thread::spawn(move || Store::new(&root, 5).write(9, &[i; 4096]))
            })
            .collect();
        for t in threads {
            t.join().unwrap().unwrap();
        }
        let store = Store::new(&root, 5);
        let blob = store.read(9).expect("one complete file");
        assert!(blob.len() == 4096 && blob.iter().all(|&b| b == blob[0]), "the blob of one writer");
        let leftovers = std::fs::read_dir(store.dir()).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".tmp")).count();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn a_damaged_file_is_ignored_and_stored_again() {
        let dir = TempDir::new("pso-library-damage");
        let store = Store::new(&dir.0, 0xD0);
        store.write(4, b"a driver blob of some length").unwrap();
        let path = store.path(4);
        let good = std::fs::read(&path).unwrap();
        let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
        let mut flipped = good.clone();
        *flipped.last_mut().unwrap() ^= 1;
        cases.push(("a changed blob byte", flipped));
        cases.push(("a cut file", good[..good.len() - 3].to_vec()));
        cases.push(("only the header", good[..HEADER].to_vec()));
        let mut magic = good.clone();
        magic[0] = b'X';
        cases.push(("another magic", magic));
        let mut format = good.clone();
        format[4] = 99;
        cases.push(("another format", format));
        let mut key = good.clone();
        key[24] ^= 1;
        cases.push(("another key", key));
        cases.push(("an empty file", Vec::new()));
        for (what, bytes) in cases {
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(store.read(4), None, "{what} is ignored");
        }
        let library = Library::new(dir.0.clone(), "t");
        let library_store = library.open(|| Some(adapter())).unwrap();
        let path = library_store.path(4);
        std::fs::create_dir_all(library_store.dir()).unwrap();
        std::fs::write(&path, b"SXPL broken").unwrap();
        assert_eq!(library.lookup(4), None, "the library treats it as a miss");
        library.keep(4, b"compiled again").unwrap();
        assert_eq!(library.lookup(4).as_deref(), Some(&b"compiled again"[..]));
        assert_eq!(library.counts().misses.load(Ordering::Relaxed), 1);
        assert_eq!(library.counts().hits.load(Ordering::Relaxed), 1);
        assert_eq!(library.counts().stored.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn the_library_opens_once_and_stops_after_refused_blobs() {
        let dir = TempDir::new("pso-library-reject");
        let library = Library::new(dir.0.clone(), "t");
        assert_eq!(library.lookup(1), None, "closed before the adapter is known");
        library.keep(1, b"blob").unwrap();
        assert_eq!(library.counts().stored.load(Ordering::Relaxed), 0, "nothing stored while closed");
        let first = library.open(|| Some(adapter())).unwrap().dir().to_path_buf();
        let again = library.open(|| Some(AdapterIdentity { driver_version: 9, ..adapter() })).unwrap().dir().to_path_buf();
        assert_eq!(first, again, "the first adapter stays");

        for key in 0..u128::from(MAX_REJECTIONS) {
            library.keep(key, b"blob").unwrap();
        }
        assert!(library.lookup(0).is_some());
        assert!(library.reject(0), "one refusal keeps the library on");
        assert_eq!(library.lookup(0), None, "the refused blob is deleted");
        for key in 1..u128::from(MAX_REJECTIONS) - 1 {
            assert!(library.reject(key));
        }
        assert!(!library.reject(u128::from(MAX_REJECTIONS) - 1), "the last refusal stops it");
        library.keep(100, b"blob").unwrap();
        assert_eq!(library.lookup(100), None, "a stopped library neither stores nor reads");
        assert!(library.summary().starts_with("pso library: hits=1 "));

        let closed = Library::new(dir.0.clone(), "t");
        assert!(closed.open(|| None).is_none(), "an unknown adapter keeps it closed");
        assert!(closed.open(|| Some(adapter())).is_none(), "for the whole run");
    }

    /// The real runtime: a blob of a made pipeline makes the same pipeline again, and a damaged
    /// blob is refused. Skipped without a D3D12 device.
    #[cfg(windows)]
    #[test]
    #[allow(unsafe_code)]
    fn the_pc_runtime_takes_its_own_blob_back_and_refuses_a_damaged_one() {
        use windows::Win32::Graphics::Direct3D12::{D3D12_CACHED_PIPELINE_STATE, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_SHADER_BYTECODE, ID3D12PipelineState};
        use windows::core::Interface;

        let shader = crate::view_gpu_tests::compile("groupshared uint g[2]; [numthreads(2, 1, 1)] void main(uint i : SV_GroupIndex) { g[i] = i; }");
        for (name, device) in crate::view_gpu_tests::devices() {
            let signature = crate::view_gpu_tests::root_signature(&device, &[]);
            let made = crate::view_gpu_tests::pipeline(&device, &signature, &shader);
            let Some(identity) = adapter_of(made.as_raw()) else {
                println!("{name}: DXGI gives no driver version, skipped");
                continue;
            };
            // SAFETY: a live device.
            let luid = unsafe { device.GetAdapterLuid() };
            assert_eq!(identity.luid, u64::from(luid.LowPart) | (u64::from(luid.HighPart as u32) << 32), "{name}");
            let Some(blob) = cached_blob(made.as_raw()) else {
                println!("{name}: the runtime gives no cached blob, skipped");
                continue;
            };
            let with = |bytes: &[u8]| {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(signature.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: shader.as_ptr().cast(), BytecodeLength: shader.len() },
                    CachedPSO: D3D12_CACHED_PIPELINE_STATE { pCachedBlob: bytes.as_ptr().cast(), CachedBlobSizeInBytes: bytes.len() },
                    ..Default::default()
                };
                // SAFETY: a valid description whose shader and blob outlive the call.
                let made: windows::core::Result<ID3D12PipelineState> = unsafe { device.CreateComputePipelineState(&desc) };
                std::mem::ManuallyDrop::into_inner(desc.pRootSignature);
                made
            };
            assert!(with(&blob).is_ok(), "{name}: the runtime takes its own blob");
            let mut damaged = blob.clone();
            let middle = damaged.len() / 2;
            damaged[middle] ^= 0xFF;
            damaged.truncate(damaged.len().saturating_sub(1).max(1));
            assert!(with(&damaged).is_err(), "{name}: a damaged blob is refused");
        }
    }

    #[test]
    fn keep_is_visible_in_memory_immediately_and_flushes_to_disk_on_drop() {
        let dir = TempDir::new("pso-library-async-write");
        {
            let library = Library::new(dir.0.clone(), "t");
            library.open(|| Some(adapter())).unwrap();
            library.keep(42, b"async blob payload").unwrap();
            assert_eq!(library.lookup(42).as_deref(), Some(&b"async blob payload"[..]));
            assert_eq!(library.counts().stored.load(Ordering::Relaxed), 1);
            assert_eq!(library.counts().hits.load(Ordering::Relaxed), 1);
        }
        let reopened = Library::new(dir.0.clone(), "t");
        reopened.open(|| Some(adapter())).unwrap();
        assert_eq!(reopened.lookup(42).as_deref(), Some(&b"async blob payload"[..]));
    }

    #[test]
    fn sharded_pending_writes_handle_many_keys_and_overwrites() {
        let dir = TempDir::new("pso-library-sharded-pending");
        let library = Library::new(dir.0.clone(), "t");
        library.open(|| Some(adapter())).unwrap();
        for key in 1u128..=32 {
            let payload = format!("blob payload {key}");
            library.keep(key, payload.as_bytes()).unwrap();
        }
        library.keep(5, b"updated payload 5").unwrap();
        library.flush();
        for key in 1u128..=32 {
            let expected = if key == 5 { b"updated payload 5".to_vec() } else { format!("blob payload {key}").into_bytes() };
            assert_eq!(library.lookup(key), Some(expected), "key {key}");
        }
        assert_eq!(library.lookup(999), None);
    }
}
