//! Looks up a replacement DXIL shader blob by content hash, for console compute shaders the PC
//! runtime cannot build as the title compiled them.
//!
//! `DECANT_D3D12X_DXIL_OVERRIDE=<dir>` (default: unset, off). At a compute pipeline create, the layer
//! hashes the shader bytecode (FNV-1a 64 over every byte of its content, not its address) and, if
//! `<dir>/<hash as 16 lower case hex digits>.dxil` exists, hands that file's bytes to the host
//! instead. One file read per create; overrides come from an offline lowering pipeline and are not
//! a hot path, so nothing here is cached.
//!
//! `DECANT_D3D12X_DXIL_OVERRIDE_MAX_BYTES=<bytes>` (default: 4 MiB; `0` = no cap) is the largest
//! override file that is read. A bigger file is not read at all (its size comes from the metadata)
//! and the lookup is a miss; the size and the cap are logged once per hash.
//!
//! The override file may already be a complete DXBC container (what the lowering tool normally
//! emits) or a bare DXIL part with no "DXBC" header: [`wrap_bare_dxil`] puts the latter in a
//! minimal valid container with a correct checksum, so the host always sees the same shape a
//! compiler would have produced. Anything that is not readable, or reads empty, is treated as a
//! miss: the caller keeps the title's own bytecode, exactly as the C `dxil_override` did on an
//! unreadable file.
//!
//! `DECANT_D3D12X_SPACE_HINTS=<file>` (default: unset, off) feeds the offline tool back: when an
//! override is refused because registers sit in another space than the title's root signature has
//! them, one line per register is appended to that file (`<hash> <class letter> <register> <old space>
//! <new space>`), which `decant_shader_dxil_cache --space-hints` reads to place the registers right.

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use crate::rts0::{RegisterClass, RootLayout, RootParamKind, dxbc_checksum, wr32};

/// FNV-1a 64, the hash the C layer's `dxil_content_hash` used.
pub fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// The file name an override for `bytes` would have in the override directory: 16 lower case hex
/// digits of [`content_hash`] and the `.dxil` extension, as the C layer named them.
pub fn file_name(bytes: &[u8]) -> String {
    file_name_for(content_hash(bytes))
}

fn file_name_for(hash: u64) -> String {
    format!("{hash:016x}.dxil")
}

/// FNV-1a 64 with the exact literal the earlier C layer used
/// (decimal `1469598103934665603`). That literal is
/// NOT the published FNV-1a 64 offset basis (`14695981039346656037`, used by [`content_hash`]
/// above): it is the canonical decimal value with its last digit dropped, a transcription slip in
/// the C source (confirmed by reading the two literals side by side: 19 digits vs the canonical
/// 20). The slip was never caught because the sealed override directory
/// (the earlier override directory) was itself generated and named with this same literal, so the C
/// layer and its own override files always agreed; `XboxBvhBuilder.dll` is the retail package's
/// own file in both the C-layer reference run and every trunk run (the launcher treats a
/// package-shipped DLL as "not replaced"), so both layers hand the same builder the same inputs.
/// Kept under its own name (not folded into `content_hash`) so a reader never mistakes it for the
/// correct algorithm; [`lookup_in`] tries both, and logs which one matched (see below) so a run
/// against the real sealed directory is itself the proof.
fn legacy_content_hash(bytes: &[u8]) -> u64 {
    let mut hash = 1_469_598_103_934_665_603u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

/// The file name an override for `bytes` would have if it were keyed by the C layer's actual
/// (non-canonical) hash, as every file in the sealed override directory in fact is.
pub fn legacy_file_name(bytes: &[u8]) -> String {
    file_name_for(legacy_content_hash(bytes))
}

/// Writes `original`'s bytes under `dir` (file name: 16 lower-case hex digits of
/// [`content_hash`], extension `.bin` so it never collides with a real `.dxil` override) for a
/// later byte-level comparison, when a directory is supplied. No-op when `dir` is `None` (the
/// default): the only reason to keep a missed shader's bytes around is to find out why the
/// override set does not recognise them (see [`dump_dir`]).
fn dump_miss_to(dir: Option<&Path>, hash: u64, original: &[u8]) {
    if let Some(dir) = dir {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(format!("{hash:016x}.bin")), original);
    }
}

/// The configured dump directory (`DECANT_D3D12X_DXIL_DUMP`), or `None` when unset (default, off).
pub fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("DECANT_D3D12X_DXIL_DUMP").map(PathBuf::from)
}

/// Wraps a bare DXIL part (no "DXBC" header) in a single part DXBC container with a correct
/// checksum. Bytes that already look like a container (start with "DXBC") are returned unchanged:
/// a malformed or truncated container is passed through as is, not reinterpreted as a bare part.
pub fn wrap_bare_dxil(part: &[u8]) -> Vec<u8> {
    if part.len() >= 4 && &part[..4] == b"DXBC" {
        return part.to_vec();
    }
    const HEADER: usize = 32 + 4; // DXBC header (32 bytes) plus one part offset word.
    let total = HEADER + 8 + part.len();
    let mut blob = vec![0u8; total];
    blob[..4].copy_from_slice(b"DXBC");
    wr32(&mut blob, 20, 1); // DXBC version
    wr32(&mut blob, 24, total as u32);
    wr32(&mut blob, 28, 1); // one part
    wr32(&mut blob, 32, HEADER as u32); // that part's offset
    blob[HEADER..HEADER + 4].copy_from_slice(b"DXIL");
    wr32(&mut blob, HEADER + 4, part.len() as u32);
    blob[HEADER + 8..].copy_from_slice(part);
    let sum = dxbc_checksum(&blob, total);
    for (i, word) in sum.iter().enumerate() {
        wr32(&mut blob, 4 + 4 * i, *word);
    }
    blob
}

/// Reads an override for `original` from `dir`. `None` (use the title's own bytecode unchanged)
/// when `original` is empty, the file does not exist, cannot be read, or reads empty. Logs a miss
/// once per distinct content hash (hash, byte length, directory, I/O error kind): a silent miss
/// left a miss unexplained when the C layer, with the same override directory, hit on
/// the same shader.
pub fn lookup_in(dir: &Path, original: &[u8]) -> Option<Vec<u8>> {
    find_in(dir, original).map(|(bytes, _)| bytes)
}

/// [`lookup_in`] plus the hash whose file name matched (canonical or legacy), which names the
/// `.json` report beside the override.
fn find_in(dir: &Path, original: &[u8]) -> Option<(Vec<u8>, u64)> {
    find_capped(dir, original, max_bytes())
}

/// [`find_in`] with the size cap given. An override file larger than `cap` bytes is not read: its
/// size is checked from the metadata first, the lookup is a miss and the refusal is logged once
/// per hash (see [`max_bytes`]).
fn find_capped(dir: &Path, original: &[u8], cap: u64) -> Option<(Vec<u8>, u64)> {
    if original.is_empty() {
        return None;
    }
    let hash = content_hash(original);
    // Try the canonical name first, then the name every file in the sealed override directory
    // actually has today (see `legacy_content_hash`): both read the same `original` bytes, so
    // trying one after the other costs one extra failed `stat` on the common case and nothing on
    // a hit.
    let mut last_err = None;
    let mut oversize = None;
    for (which, candidate) in [("canonical", hash), ("legacy", legacy_content_hash(original))] {
        let path = dir.join(file_name_for(candidate));
        match std::fs::metadata(&path) {
            Ok(meta) if meta.len() > cap => {
                oversize = Some(meta.len());
                continue;
            }
            Ok(_) => {}
            Err(err) => {
                last_err = Some(err.kind());
                continue;
            }
        }
        match std::fs::read(&path) {
            Ok(data) if !data.is_empty() => {
                #[cfg(windows)]
                if which == "legacy" {
                    crate::log::xs_log!(
                        "DXIL override hit via the legacy (non-canonical) hash only, hash={candidate:016x}, \
                         canonical hash={hash:016x} did not match any file: confirms the sealed override \
                         directory is keyed with the C layer's offset-basis typo, not FNV-1a's published one"
                    );
                }
                #[cfg(not(windows))]
                let _ = which;
                return Some((wrap_bare_dxil(&data), candidate));
            }
            Ok(_) => {}
            Err(err) => last_err = Some(err.kind()),
        }
    }
    dump_miss_to(dump_dir().as_deref(), hash, original);
    if let Some(len) = oversize {
        log_gap_once(hash, &format!("file size {len} bytes is over the cap of {cap} bytes (DECANT_D3D12X_DXIL_OVERRIDE_MAX_BYTES)"));
        return None;
    }
    log_miss_once(hash, original.len(), Some(dir), last_err);
    None
}

/// Distinct content hashes already logged as a miss (so a hot path calling this every frame does
/// not flood the log with the same line).
static LOGGED_MISSES: LazyLock<Mutex<HashSet<u64>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// Logs a DXIL-override miss the first time this content hash is seen; returns whether this call
/// is the one that logged it (`false` on every later call for the same hash). The log line itself
/// (`crate::log`) exists only on Windows (`d3d12_xs.dll`'s own log); this module stays buildable
/// and testable on any host, so the dedupe tracking runs everywhere and only the line it would
/// print is Windows-only.
fn log_miss_once(hash: u64, len: usize, dir: Option<&Path>, err: Option<io::ErrorKind>) -> bool {
    let newly_logged = LOGGED_MISSES.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(hash);
    #[cfg(windows)]
    if newly_logged {
        crate::log::xs_log!(
            "DXIL override miss, hash={hash:016x}, size {len}, dir {}, {}",
            dir.map_or_else(|| "unset".to_string(), |d| d.display().to_string()),
            err.map_or_else(|| "no such file".to_string(), |kind| format!("{kind:?}"))
        );
    }
    #[cfg(not(windows))]
    {
        let _ = (len, dir, err);
    }
    newly_logged
}

/// The configured override directory (`DECANT_D3D12X_DXIL_OVERRIDE`), or `None` when unset (default,
/// overrides off).
pub fn configured_dir() -> Option<PathBuf> {
    std::env::var_os("DECANT_D3D12X_DXIL_OVERRIDE").map(PathBuf::from)
}

/// Default largest override file, in bytes (4 MiB). Overrides of 30 to 47 MiB made the Vulkan
/// driver's shader compiler allocate tens of GiB and the title was killed by the OOM killer.
/// The files that size are rare (28 of 10,394 are at least 4 MiB).
const DEFAULT_MAX_BYTES: u64 = 4 << 20;

/// The size cap from the raw `DECANT_D3D12X_DXIL_OVERRIDE_MAX_BYTES` value (`None` when unset): a
/// decimal byte count, `0` for no cap (`u64::MAX`). Unset or not a number gives the default.
fn size_cap_from(raw: Option<&str>) -> u64 {
    match raw.and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(0) => u64::MAX,
        Some(bytes) => bytes,
        None => DEFAULT_MAX_BYTES,
    }
}

/// The largest override file that is read, in bytes (`DECANT_D3D12X_DXIL_OVERRIDE_MAX_BYTES`, read the
/// way `DECANT_D3D12X_TITLE_WINDOW_GIB` is read).
fn max_bytes() -> u64 {
    size_cap_from(std::env::var("DECANT_D3D12X_DXIL_OVERRIDE_MAX_BYTES").ok().as_deref())
}

/// Looks up an override for `original` under the configured directory. `None` (pass the original
/// bytecode through unchanged) when overrides are off or there is no match. Logs a miss (once per
/// distinct hash) when overrides are off too, same as [`lookup_in`]'s miss log, dir "unset".
pub fn lookup(original: &[u8]) -> Option<Vec<u8>> {
    match configured_dir() {
        Some(dir) => lookup_in(&dir, original),
        None => {
            if !original.is_empty() {
                let hash = content_hash(original);
                dump_miss_to(dump_dir().as_deref(), hash, original);
                log_miss_once(hash, original.len(), None, None);
            }
            None
        }
    }
}

/// `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`.
const FLAG_RESOURCE_HEAP: u32 = 0x400;
/// `D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED`.
const FLAG_SAMPLER_HEAP: u32 = 0x800;

/// Class of a register in a binding report. Written lower case in the json.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReportClass {
    /// `t`.
    Srv,
    /// `u`.
    Uav,
    /// `b`.
    Cbv,
    /// `s`.
    Sampler,
}

impl ReportClass {
    fn to_rts0(self) -> RegisterClass {
        match self {
            ReportClass::Srv => RegisterClass::Srv,
            ReportClass::Uav => RegisterClass::Uav,
            ReportClass::Cbv => RegisterClass::Cbv,
            ReportClass::Sampler => RegisterClass::Sampler,
        }
    }

    fn letter(self) -> char {
        match self {
            ReportClass::Srv => 't',
            ReportClass::Uav => 'u',
            ReportClass::Cbv => 'b',
            ReportClass::Sampler => 's',
        }
    }
}

/// One register a translated shader declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct ReportRegister {
    /// Register class.
    pub class: ReportClass,
    /// Register number.
    pub register: u32,
    /// Register space (0 when left out).
    #[serde(default)]
    pub space: u32,
}

/// The binding report `<hash>.json` the offline tool writes beside `<hash>.dxil`:
///
/// ```json
/// {"registers": [{"class": "srv", "register": 188, "space": 0},
///                {"class": "cbv", "register": 27}],
///  "resource_heap": true, "sampler_heap": false}
/// ```
///
/// `class` is `srv`, `uav`, `cbv` or `sampler`. `space` defaults to 0. `resource_heap` is true when
/// the shader indexes the CBV/SRV/UAV heap directly (SM 6.6 `createHandleFromHeap`), which needs
/// the root signature flag `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`, and `sampler_heap` likewise for
/// `SAMPLER_HEAP_DIRECTLY_INDEXED`. Every field may be left out, and unknown fields are ignored.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
pub struct BindingReport {
    /// Registers read through the root signature.
    #[serde(default)]
    pub registers: Vec<ReportRegister>,
    /// The shader indexes the resource heap directly.
    #[serde(default)]
    pub resource_heap: bool,
    /// The shader indexes the sampler heap directly.
    #[serde(default)]
    pub sampler_heap: bool,
    /// Present when the shader reads the user data mirror (`"user_data":{"space":15,"register":1,
    /// "dwords":32,"used":[8,9]}`): the root signature needs the mirror parameter.
    #[serde(default)]
    pub user_data: Option<UserDataReport>,
    /// Present when the shader reads memory by device address (`"window":{"space":14,"uav_register":0,
    /// "base_register":0}`): the root signature needs the window UAV and its base address CBV.
    #[serde(default)]
    pub window: Option<WindowReport>,
    /// The translator's own buffers the shader uses (`[{"role":"page_table","class":"srv","register":0,
    /// "space":13}]`): the root signature needs the page table SRV and the service UAVs.
    #[serde(default)]
    pub service: Vec<ServiceReport>,
    /// What the shader reads at each offset of its console tables, from the resource heap through the
    /// user data mirror (`[{"dword":9,"slot":5,"class":"srv","kind":"raw_buffer"}]`), for the kind
    /// check (`kind_check`). Left out by older builds of the override.
    #[serde(default)]
    pub table_views: Vec<TableViewReport>,
}

/// One descriptor of a console table a translated shader reads.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct TableViewReport {
    /// The mirror dword that holds the heap index of the table.
    pub dword: u32,
    /// The slot in the table (the offset in 32 byte descriptors).
    pub slot: u32,
    /// `srv`, `uav` or `cbv`.
    pub class: String,
    /// `typed_buffer`, `raw_buffer`, `structured_buffer`, `cbuffer`, `texture1d`, `texture2d`, `texture3d`,
    /// `texture_cube`, `texture1d_array`, `texture2d_array` or `texture_cube_array`.
    pub kind: String,
}

/// The device memory window a translated shader reads, as the importer reports it.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
pub struct WindowReport {
    /// Register space of the window (14).
    #[serde(default)]
    pub space: u32,
    /// Register of the window UAV (0).
    #[serde(default)]
    pub uav_register: u32,
    /// Register of the base address CBV (0).
    #[serde(default)]
    pub base_register: u32,
}

/// One of the translator's own buffers a shader uses.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct ServiceReport {
    /// What the buffer is for (`page_table`, `pages_touched`, `buffer_55`, `buffer_56`).
    #[serde(default)]
    pub role: String,
    /// `srv` or `uav`.
    pub class: ReportClass,
    /// Register in the space.
    pub register: u32,
    /// Register space (13).
    #[serde(default)]
    pub space: u32,
}

/// The user data mirror a translated shader reads, as the importer reports it.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
pub struct UserDataReport {
    /// Register space of the constant buffer (15).
    #[serde(default)]
    pub space: u32,
    /// Register of the constant buffer (1).
    #[serde(default)]
    pub register: u32,
    /// Size of the mirror in dwords (32).
    #[serde(default)]
    pub dwords: u32,
    /// The dwords (user SGPRs) the shader reads.
    #[serde(default)]
    pub used: Vec<u32>,
}

/// Whether `layout` has `reg` for a stage with shader visibility `stage` (0 for compute, where only
/// parameters visible to all stages count).
fn covers(layout: &RootLayout, reg: ReportRegister, stage: u32) -> bool {
    let class = reg.class.to_rts0();
    let visible = |v: u32| v == 0 || v == stage;
    if class == RegisterClass::Sampler && layout.static_samplers.iter().any(|s| s.register == reg.register && s.space == reg.space && visible(s.visibility)) {
        return true;
    }
    layout.params.iter().filter(|p| visible(p.visibility)).any(|p| match &p.kind {
        RootParamKind::Table(ranges) => ranges.iter().any(|r| {
            r.class == class && r.space == reg.space && reg.register >= r.base && (r.count == u32::MAX || u64::from(reg.register) < u64::from(r.base) + u64::from(r.count))
        }),
        RootParamKind::Constants { register, space, .. } => class == RegisterClass::Cbv && *register == reg.register && *space == reg.space,
        RootParamKind::Descriptor { class: c, register, space } => *c == class && *register == reg.register && *space == reg.space,
    })
}

/// The first thing a translated shader needs that `layout` does not provide, as a sentence for the
/// log, or `None` when everything is covered. `stage` is the `D3D12_SHADER_VISIBILITY` of the shader
/// stage (0 for compute).
pub fn first_gap(report: &BindingReport, layout: &RootLayout, stage: u32) -> Option<String> {
    for reg in &report.registers {
        if !covers(layout, *reg, stage) {
            // Where the register is instead, if anywhere: another space, or a parameter that is
            // not visible to this stage, says which side of the bridge is off.
            let elsewhere: Vec<String> = (0..16u32)
                .flat_map(|space| [0u32, 1, 2, 3, 4, 5, 6, 7].map(move |v| (space, v)))
                .filter(|&(space, v)| (space, v) != (reg.space, stage) && covers(layout, ReportRegister { space, ..*reg }, v))
                .map(|(space, v)| format!("space {space} visibility {v}"))
                .take(3)
                .collect();
            let hint = if elsewhere.is_empty() { "not in any space or stage".to_string() } else { format!("it is there in {}", elsewhere.join(", ")) };
            return Some(format!("register {}{} space {} is not in the root signature for stage {stage} ({hint})", reg.class.letter(), reg.register, reg.space));
        }
    }
    if let Some(ud) = &report.user_data {
        if layout.mirror_index().is_none() {
            return Some("the shader reads the user data mirror but the root signature has no mirror parameter".to_string());
        }
        if (ud.space, ud.register) != (crate::rts0::MIRROR_SPACE, crate::rts0::MIRROR_REGISTER) || ud.dwords as usize > crate::user_mirror::DWORDS {
            return Some(format!("the user data mirror of the shader (b{} space {}, {} dwords) is not the layer's", ud.register, ud.space, ud.dwords));
        }
    }
    if let Some(w) = &report.window {
        if (w.space, w.uav_register, w.base_register) != (crate::rts0::WINDOW_SPACE, 0, 0) {
            return Some(format!("the device memory window of the shader (u{} and b{} of space {}) is not the layer's", w.uav_register, w.base_register, w.space));
        }
        if layout.window_uav_index().is_none() || layout.window_base_index().is_none() {
            return Some("the shader reads device memory through the window but the root signature has no window parameters".to_string());
        }
    }
    for s in &report.service {
        let (index, name) = match (s.class, s.register) {
            (ReportClass::Srv, 0) => (layout.service_indices()[0], "t0"),
            (ReportClass::Uav, r @ 0..=2) => (layout.service_indices()[1 + r as usize], ["u0", "u1", "u2"][r as usize]),
            _ => return Some(format!("the service buffer {}{} of space {} is not the layer's", s.class.letter(), s.register, s.space)),
        };
        if s.space != crate::rts0::SERVICE_SPACE {
            return Some(format!("the service buffer {name} of space {} is not the layer's", s.space));
        }
        if index.is_none() {
            return Some(format!("the shader uses the service buffer {name} ({}) but the root signature has no service parameters", s.role));
        }
    }
    if report.resource_heap && layout.flags & FLAG_RESOURCE_HEAP == 0 {
        return Some("the shader indexes the resource heap but the root signature lacks CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED".to_string());
    }
    if report.sampler_heap && layout.flags & FLAG_SAMPLER_HEAP == 0 {
        return Some("the shader indexes the sampler heap but the root signature lacks SAMPLER_HEAP_DIRECTLY_INDEXED".to_string());
    }
    None
}

/// One register of a refused override that the root signature has in another space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpaceHint {
    /// Register class.
    pub class: ReportClass,
    /// Register number.
    pub register: u32,
    /// The space the override has it in.
    pub old: u32,
    /// The only space the root signature has it in for this stage.
    pub new: u32,
}

impl SpaceHint {
    /// The hint as one line of the hints file: `<hash> <class letter> <register> <old space> <new space>`,
    /// the hash as 16 lower case hex digits.
    pub fn line(&self, hash: u64) -> String {
        format!("{hash:016x} {} {} {} {}", self.class.letter(), self.register, self.old, self.new)
    }
}

/// The spaces the title's own registers use. The layer's own spaces (service, window, mirror) are
/// never moved, they are the layer's parameters and not a guess of the translator.
const FIRST_LAYER_SPACE: u32 = crate::rts0::SERVICE_SPACE;

/// For every register of `report` that `layout` lacks for `stage`: the space that has it, when that
/// is exactly one (a parameter visible to all stages or to this one). A register that no space has,
/// or that several have, gives no hint, because there is nothing certain to move it to.
pub fn space_hints(report: &BindingReport, layout: &RootLayout, stage: u32) -> Vec<SpaceHint> {
    let mut out = Vec::new();
    for reg in &report.registers {
        if reg.space >= FIRST_LAYER_SPACE || covers(layout, *reg, stage) {
            continue;
        }
        let spaces: Vec<u32> = (0..FIRST_LAYER_SPACE).filter(|&space| covers(layout, ReportRegister { space, ..*reg }, stage)).collect();
        if let [new] = spaces[..] {
            out.push(SpaceHint { class: reg.class, register: reg.register, old: reg.space, new });
        }
    }
    out
}

/// The hints file (`DECANT_D3D12X_SPACE_HINTS`), or `None` when unset (default, off).
fn hints_path() -> Option<PathBuf> {
    std::env::var_os("DECANT_D3D12X_SPACE_HINTS").map(PathBuf::from)
}

/// What the run wrote so far: the lines (with the root signature they came from, so a hint that
/// several root signatures give is written once per root signature and the cache tool can count
/// them) and the space each `(hash, class, register)` was given.
#[derive(Default)]
struct HintLog {
    seen: HashSet<(String, u64)>,
    spaces: std::collections::HashMap<(u64, char, u32), Vec<u32>>,
}

static HINTS_WRITTEN: LazyLock<Mutex<HintLog>> = LazyLock::new(|| Mutex::new(HintLog::default()));

/// Appends the hints for the refused override `hash` to the hints file, when one is configured.
/// `layout_key` tells root signatures apart. Returns the lines newly written (for the tests).
fn write_space_hints(path: Option<&Path>, hash: u64, hints: &[SpaceHint], layout_key: u64) -> Vec<String> {
    let Some(path) = path else { return Vec::new() };
    let mut log = HINTS_WRITTEN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut fresh = Vec::new();
    for hint in hints {
        let line = hint.line(hash);
        if !log.seen.insert((line.clone(), layout_key)) {
            continue;
        }
        let spaces = log.spaces.entry((hash, hint.class.letter(), hint.register)).or_default();
        if !spaces.is_empty() && !spaces.contains(&hint.new) {
            #[cfg(windows)]
            crate::log::xs_log!(
                "DXIL override {hash:016x}: {}{} is in space {} in this root signature but in {spaces:?} in another, both hints written, the cache tool takes the most frequent",
                hint.class.letter(),
                hint.register,
                hint.new
            );
        }
        spaces.push(hint.new);
        fresh.push(line);
    }
    if !fresh.is_empty() {
        use std::io::Write;
        let text = fresh.iter().map(|l| format!("{l}
")).collect::<String>();
        let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(|mut f| f.write_all(text.as_bytes()));
    }
    fresh
}

/// Hashes whose override was refused because of a gap or an unreadable report, logged once each.
static LOGGED_GAPS: LazyLock<Mutex<HashSet<u64>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// Logs `what` the first time `hash` is refused. Returns whether this call was the first.
fn log_gap_once(hash: u64, what: &str) -> bool {
    let first = LOGGED_GAPS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(hash);
    #[cfg(windows)]
    if first {
        crate::log::xs_log!("DXIL override refused, hash={hash:016x}: {what}, keeping the title's bytecode");
    }
    #[cfg(not(windows))]
    let _ = what;
    first
}

/// What became of the DXIL override of one shader stage, for the log line of a refused pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Overrides are off (`DECANT_D3D12X_DXIL_OVERRIDE` unset).
    Off,
    /// The override replaced the title's shader.
    Hit,
    /// No override file for the hash.
    Miss,
    /// An override file exists but was not used, for the reason given.
    Refused(String),
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Off => f.write_str("overrides off"),
            Outcome::Hit => f.write_str("hit"),
            Outcome::Miss => f.write_str("miss"),
            Outcome::Refused(why) => write!(f, "refused ({why})"),
        }
    }
}

/// Like [`lookup_in`], and when a `<hash>.json` binding report ([`BindingReport`]) sits beside the
/// override, checks it against `layout` (the root signature of the pipeline, `None` when unknown) for
/// a stage with visibility `stage`. A gap logs once per hash and gives `None`, so the caller keeps
/// the title's bytecode as for a miss. No report, an unreadable one or an unknown root signature
/// let the override through and the runtime decide, as before.
pub fn lookup_checked_in(dir: &Path, original: &[u8], layout: Option<&RootLayout>, stage: u32) -> Option<Vec<u8>> {
    checked_in(dir, original, layout, stage).0
}

/// [`lookup_checked_in`] plus what became of the override.
fn checked_in(dir: &Path, original: &[u8], layout: Option<&RootLayout>, stage: u32) -> (Option<Vec<u8>>, Outcome) {
    let Some((bytes, hash)) = find_in(dir, original) else {
        return (None, Outcome::Miss);
    };
    let Ok(text) = std::fs::read(dir.join(format!("{hash:016x}.json"))) else {
        return (Some(bytes), Outcome::Hit);
    };
    let report: BindingReport = match serde_json::from_slice(&text) {
        Ok(report) => report,
        Err(err) => {
            log_gap_once(hash, &format!("the binding report is not readable ({err}), not checked"));
            return (Some(bytes), Outcome::Hit);
        }
    };
    crate::kind_check::override_report(content_hash(original), hash, &report.table_views);
    let Some(layout) = layout else {
        return (Some(bytes), Outcome::Hit);
    };
    if let Some(gap) = first_gap(&report, layout, stage) {
        log_gap_once(hash, &gap);
        if let Some(path) = hints_path() {
            let hints = space_hints(&report, layout, stage);
            if !hints.is_empty() {
                write_space_hints(Some(&path), hash, &hints, content_hash(format!("{layout:?}").as_bytes()));
            }
        }
        return (None, Outcome::Refused(gap));
    }
    (Some(bytes), Outcome::Hit)
}

/// Layouts of the root signatures the title created (device slot 16), by the address of the PC
/// object. An entry stays after the object is released: an address that is reused is overwritten at
/// the next create, and only root signature objects are looked up.
static ROOT_LAYOUTS: LazyLock<Mutex<std::collections::HashMap<usize, RootLayout>>> = LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

/// Remembers the layout of the root signature object at `object`.
pub fn remember_root_layout(object: usize, layout: RootLayout) {
    ROOT_LAYOUTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(object, layout);
}

/// The layout of the root signature object at `object`, when the title created it through slot 16.
pub fn root_layout(object: usize) -> Option<RootLayout> {
    ROOT_LAYOUTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&object).cloned()
}

/// The override for the shader `original` of a pipeline whose root signature object is `root_sig`,
/// for a stage with visibility `stage` (0 for compute). Same as [`lookup`] plus the coverage check
/// of [`lookup_checked_in`].
pub fn lookup_for_pipeline(original: &[u8], root_sig: usize, stage: u32) -> Option<Vec<u8>> {
    lookup_for_pipeline_outcome(original, root_sig, stage).0
}

/// [`lookup_for_pipeline`] plus what became of the override.
pub fn lookup_for_pipeline_outcome(original: &[u8], root_sig: usize, stage: u32) -> (Option<Vec<u8>>, Outcome) {
    match configured_dir() {
        Some(dir) => {
            let layout = root_layout(root_sig);
            checked_in(&dir, original, layout.as_ref(), stage)
        }
        None => (lookup(original), Outcome::Off),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory for one test, removed by its own `Drop`.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> TempDir {
            let dir = std::env::temp_dir().join(format!("d3d12xs-dxil-override-test-{label}-{}-{:?}", std::process::id(), std::thread::current().id()));
            std::fs::create_dir_all(&dir).expect("a scratch directory");
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_bare_dxil_part_is_wrapped_in_a_container_with_a_correct_checksum() {
        let part = b"not real DXIL bitcode, just synthetic bytes for the test".to_vec();
        let wrapped = wrap_bare_dxil(&part);
        assert_eq!(&wrapped[..4], b"DXBC");
        assert_eq!(crate::rts0::rd32(&wrapped, 24) as usize, wrapped.len());
        let sum = dxbc_checksum(&wrapped, wrapped.len());
        assert!((0..4).all(|i| crate::rts0::rd32(&wrapped, 4 + 4 * i) == sum[i]), "checksum not written correctly");
        assert_eq!(&wrapped[wrapped.len() - part.len()..], part.as_slice(), "the DXIL bytes themselves are untouched");
    }

    #[test]
    fn a_container_that_already_has_a_dxbc_header_is_not_rewrapped() {
        let container = b"DXBC and whatever bytes follow, a stand-in for a real container".to_vec();
        assert_eq!(wrap_bare_dxil(&container), container);
    }

    #[test]
    fn a_hit_hands_back_the_override_file_contents_wrapped() {
        let dir = TempDir::new("hit");
        let original = b"synthetic original console shader bytes".to_vec();
        std::fs::write(dir.0.join(file_name(&original)), b"synthetic override bitcode").unwrap();
        let got = lookup_in(&dir.0, &original).expect("a file exists for this hash");
        assert_eq!(&got[got.len() - "synthetic override bitcode".len()..], b"synthetic override bitcode");
    }

    #[test]
    fn a_miss_passes_the_original_bytecode_through_unchanged() {
        let dir = TempDir::new("miss");
        assert_eq!(lookup_in(&dir.0, b"nobody prepared an override for this synthetic blob"), None);
    }

    #[test]
    fn a_malformed_override_file_is_treated_as_a_miss() {
        let dir = TempDir::new("malformed");
        let original = b"another synthetic original blob".to_vec();
        std::fs::write(dir.0.join(file_name(&original)), b"").unwrap(); // empty: unreadable in spirit
        assert_eq!(lookup_in(&dir.0, &original), None, "an empty override file must not be handed to the host");
    }

    #[test]
    fn empty_bytecode_never_looks_up_an_override() {
        let dir = TempDir::new("empty-input");
        assert_eq!(lookup_in(&dir.0, &[]), None);
    }

    #[test]
    fn the_file_name_is_sixteen_lower_case_hex_digits_and_the_dxil_extension() {
        let name = file_name(b"anything");
        assert_eq!(name.len(), 16 + 5);
        assert!(name.ends_with(".dxil"));
        assert!(name[..16].chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn legacy_content_hash_is_not_the_canonical_fnv_offset_basis() {
        // The two offset bases differ only because the C layer's literal is the canonical decimal
        // value with its last digit dropped (19 digits vs 20) - confirmed directly on the empty
        // input, where the hash IS the offset basis (the loop never runs).
        assert_eq!(content_hash(&[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(legacy_content_hash(&[]), 1_469_598_103_934_665_603);
        assert_ne!(content_hash(&[]), legacy_content_hash(&[]));
    }

    #[test]
    fn a_shader_present_only_under_the_legacy_hash_still_hits() {
        // Reproduces the sealed override directory as it actually is: every file there is named
        // with `legacy_content_hash`, not `content_hash` (see the module docs of this file).
        let dir = TempDir::new("legacy-hit");
        let original = b"stand-in for the real 5136-byte slot 62 compute shader bytes".to_vec();
        std::fs::write(dir.0.join(legacy_file_name(&original)), b"legacy-keyed override bitcode").unwrap();
        let got = lookup_in(&dir.0, &original).expect("the legacy-named file must still be found");
        assert_eq!(&got[got.len() - "legacy-keyed override bitcode".len()..], b"legacy-keyed override bitcode");
    }

    #[test]
    fn the_canonical_name_is_preferred_when_both_exist() {
        let dir = TempDir::new("both-present");
        let original = b"a shader with both a canonical and a legacy override on disk".to_vec();
        std::fs::write(dir.0.join(file_name(&original)), b"canonical bitcode").unwrap();
        std::fs::write(dir.0.join(legacy_file_name(&original)), b"legacy bitcode").unwrap();
        let got = lookup_in(&dir.0, &original).expect("a file exists for this hash");
        assert_eq!(&got[got.len() - "canonical bitcode".len()..], b"canonical bitcode");
    }

    #[test]
    fn dump_miss_to_writes_the_original_bytes_only_when_a_directory_is_given() {
        let dir = TempDir::new("dump");
        let original = b"bytes a miss would dump for offline comparison".to_vec();
        let hash = content_hash(&original);
        dump_miss_to(None, hash, &original);
        assert!(!dir.0.join(format!("{hash:016x}.bin")).exists(), "no dir given, nothing written anywhere");
        dump_miss_to(Some(&dir.0), hash, &original);
        let dumped = std::fs::read(dir.0.join(format!("{hash:016x}.bin"))).expect("dump_miss_to must write the file");
        assert_eq!(dumped, original);
    }

    use crate::rts0::{RootParam, RootRange, StaticSampler};

    /// A layout with table t0..t3 (space 0) and unbounded u4.. (space 1), root CBV b2, constants b5,
    /// root SRV t9 visible to the pixel stage only, and static sampler s1.
    fn layout(flags: u32) -> RootLayout {
        let range = |class, base, count, space| RootRange { class, count, base, space, flags: 0, offset: 0 };
        RootLayout {
            version: 2,
            flags,
            params: vec![
                RootParam { kind: RootParamKind::Table(vec![range(RegisterClass::Srv, 0, 4, 0), range(RegisterClass::Uav, 4, u32::MAX, 1)]), visibility: 0 },
                RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 2, space: 0 }, visibility: 0 },
                RootParam { kind: RootParamKind::Constants { register: 5, space: 0, count: 4 }, visibility: 0 },
                RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Srv, register: 9, space: 0 }, visibility: 5 },
            ],
            static_samplers: vec![StaticSampler { register: 1, space: 0, visibility: 0 }],
        }
    }

    fn report(json: &str) -> BindingReport {
        serde_json::from_str(json).expect("a valid report")
    }

    #[test]
    fn registers_inside_the_layout_are_covered() {
        let r = report(
            r#"{"registers":[{"class":"srv","register":3},{"class":"uav","register":4000,"space":1},{"class":"cbv","register":2},
                {"class":"cbv","register":5},{"class":"sampler","register":1},{"class":"srv","register":9}]}"#,
        );
        assert_eq!(first_gap(&r, &layout(0), 5), None);
    }

    #[test]
    fn a_missing_register_is_named() {
        let l = layout(0);
        let gap = |json: &str, stage| first_gap(&report(json), &l, stage);
        assert_eq!(gap(r#"{"registers":[{"class":"srv","register":4}]}"#, 0).as_deref(), Some("register t4 space 0 is not in the root signature for stage 0 (not in any space or stage)"));
        assert!(gap(r#"{"registers":[{"class":"uav","register":4}]}"#, 0).is_some(), "uav range is in space 1");
        assert!(gap(r#"{"registers":[{"class":"uav","register":3,"space":1}]}"#, 0).is_some(), "below the base");
        assert!(gap(r#"{"registers":[{"class":"cbv","register":3}]}"#, 0).is_some());
        assert!(gap(r#"{"registers":[{"class":"srv","register":2}]}"#, 0).is_none(), "the table range");
        assert!(gap(r#"{"registers":[{"class":"sampler","register":2}]}"#, 0).is_some());
        assert!(gap(r#"{"registers":[{"class":"cbv","register":0}]}"#, 0).is_some(), "a cbv is not an srv table entry");
    }

    #[test]
    fn shader_visibility_is_honoured() {
        let l = layout(0);
        let r = report(r#"{"registers":[{"class":"srv","register":9}]}"#);
        assert_eq!(first_gap(&r, &l, 5), None, "visible to the pixel stage");
        assert!(first_gap(&r, &l, 1).is_some(), "not visible to the vertex stage");
        assert!(first_gap(&r, &l, 0).is_some(), "not visible to compute");
    }

    #[test]
    fn heap_use_needs_the_root_signature_flags() {
        let r = report(r#"{"resource_heap":true,"sampler_heap":true}"#);
        assert!(first_gap(&r, &layout(0), 0).unwrap().contains("CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED"));
        assert!(first_gap(&r, &layout(0x400), 0).unwrap().contains("SAMPLER_HEAP_DIRECTLY_INDEXED"));
        assert_eq!(first_gap(&r, &layout(0xC00), 0), None);
        assert_eq!(first_gap(&report("{}"), &layout(0), 0), None, "an empty report needs nothing");
    }

    #[test]
    fn a_shader_that_reads_the_mirror_needs_the_mirror_parameter() {
        let r = report(r#"{"registers":[],"user_data":{"space":15,"register":1,"dwords":32,"used":[8,9]}}"#);
        assert_eq!(r.user_data.as_ref().map(|u| (u.dwords, u.used.clone())), Some((32, vec![8, 9])));
        assert!(first_gap(&r, &layout(0), 0).unwrap().contains("no mirror parameter"));
        let mut with = layout(0);
        with.params.push(RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 });
        assert_eq!(first_gap(&r, &with, 0), None);
        let odd = report(r#"{"user_data":{"space":15,"register":2,"dwords":32}}"#);
        assert!(first_gap(&odd, &with, 0).unwrap().contains("not the layer's"));
        assert_eq!(first_gap(&report("{}"), &layout(0), 0), None, "a report without the field needs no mirror");
    }

    fn desc_param(class: RegisterClass, register: u32, space: u32) -> RootParam {
        RootParam { kind: RootParamKind::Descriptor { class, register, space }, visibility: 0 }
    }

    const WINDOW_JSON: &str = r#"{"registers":[{"class":"uav","register":0,"space":14},{"class":"cbv","register":0,"space":14},
        {"class":"srv","register":0,"space":13},{"class":"uav","register":1,"space":13}],
        "window":{"space":14,"uav_register":0,"base_register":0},
        "service":[{"role":"page_table","class":"srv","register":0,"space":13},{"role":"buffer_55","class":"uav","register":1,"space":13}]}"#;

    #[test]
    fn a_shader_that_reads_device_memory_needs_the_window_and_service_parameters() {
        let r = report(WINDOW_JSON);
        assert_eq!(r.window, Some(WindowReport { space: 14, uav_register: 0, base_register: 0 }));
        assert_eq!(r.service.len(), 2);
        let bare = layout(0);
        assert!(first_gap(&r, &bare, 0).is_some(), "the registers are not in the root signature");
        // the window and service parameters the layer appends cover the shader, in a compute stage too
        let mut with = layout(0);
        with.params.extend([
            desc_param(RegisterClass::Uav, 0, 14),
            desc_param(RegisterClass::Cbv, 0, 14),
            desc_param(RegisterClass::Srv, 0, 13),
            desc_param(RegisterClass::Uav, 0, 13),
            desc_param(RegisterClass::Uav, 1, 13),
            desc_param(RegisterClass::Uav, 2, 13),
        ]);
        assert_eq!(first_gap(&r, &with, 0), None);
        assert_eq!(first_gap(&r, &with, 5), None);
        // without the report fields nothing is needed, and the named reasons are given
        let only_window = report(r#"{"window":{"space":14,"uav_register":0,"base_register":0}}"#);
        assert!(first_gap(&only_window, &bare, 0).unwrap().contains("no window parameters"));
        assert_eq!(first_gap(&only_window, &with, 0), None);
        let only_service = report(r#"{"service":[{"role":"pages_touched","class":"uav","register":0,"space":13}]}"#);
        assert!(first_gap(&only_service, &bare, 0).unwrap().contains("no service parameters"));
        // a window without its base address parameter is not enough
        let mut half = layout(0);
        half.params.push(desc_param(RegisterClass::Uav, 0, 14));
        assert!(first_gap(&only_window, &half, 0).is_some());
        // buffers or spaces the layer does not make are named
        let odd = report(r#"{"window":{"space":14,"uav_register":1,"base_register":0}}"#);
        assert!(first_gap(&odd, &with, 0).unwrap().contains("not the layer's"));
        let odd = report(r#"{"service":[{"role":"x","class":"uav","register":3,"space":13}]}"#);
        assert!(first_gap(&odd, &with, 0).unwrap().contains("not the layer's"));
        let odd = report(r#"{"service":[{"role":"x","class":"srv","register":0,"space":12}]}"#);
        assert!(first_gap(&odd, &with, 0).unwrap().contains("not the layer's"));
        assert_eq!(first_gap(&report("{}"), &bare, 0), None, "a report without the fields needs none");
    }

    #[test]
    fn a_register_in_one_other_space_gives_a_hint() {
        let l = layout(0);
        // t1 is in the space 0 table, u4000 in the space 1 range, b2 a root CBV of space 0, t9 is pixel only.
        let r = report(
            r#"{"registers":[{"class":"srv","register":1,"space":1},{"class":"uav","register":4000,"space":0},{"class":"cbv","register":2,"space":1},
                {"class":"srv","register":9,"space":1},{"class":"srv","register":77,"space":1},{"class":"uav","register":4,"space":1}]}"#,
        );
        let hint = |class, register, old, new| SpaceHint { class, register, old, new };
        assert_eq!(
            space_hints(&r, &l, 5),
            vec![hint(ReportClass::Srv, 1, 1, 0), hint(ReportClass::Uav, 4000, 0, 1), hint(ReportClass::Cbv, 2, 1, 0), hint(ReportClass::Srv, 9, 1, 0)],
            "t77 is nowhere, u4 is covered"
        );
        assert_eq!(space_hints(&r, &l, 1).len(), 3, "t9 is not visible to the vertex stage");
        assert_eq!(hint(ReportClass::Cbv, 2, 1, 0).line(0xAB), "00000000000000ab b 2 1 0");
    }

    #[test]
    fn a_register_in_two_spaces_and_the_layer_spaces_give_no_hint() {
        let mut l = layout(0);
        l.params.push(desc_param(RegisterClass::Cbv, 2, 3));
        let r = report(r#"{"registers":[{"class":"cbv","register":2,"space":1},{"class":"uav","register":0,"space":14}]}"#);
        assert_eq!(space_hints(&r, &l, 0), vec![], "b2 is in spaces 0 and 3, u0 of space 14 is the layer's");
    }

    #[test]
    fn hints_are_appended_once_per_root_signature_and_only_when_a_file_is_given() {
        let dir = TempDir::new("hints");
        let path = dir.0.join("hints.txt");
        let hint = SpaceHint { class: ReportClass::Srv, register: 7, old: 1, new: 0 };
        let other = SpaceHint { new: 2, ..hint };
        assert!(write_space_hints(None, 0x11, &[hint], 1).is_empty());
        assert!(!path.exists());
        assert_eq!(write_space_hints(Some(&path), 0x11, &[hint], 1), vec![hint.line(0x11)]);
        assert!(write_space_hints(Some(&path), 0x11, &[hint], 1).is_empty(), "same hint, same root signature");
        assert_eq!(write_space_hints(Some(&path), 0x11, &[hint], 2).len(), 1, "another root signature counts again");
        assert_eq!(write_space_hints(Some(&path), 0x11, &[other], 3).len(), 1, "a conflicting space is written too");
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().collect::<Vec<_>>(), [hint.line(0x11), hint.line(0x11), other.line(0x11)]);
    }

    #[test]
    fn a_gap_keeps_the_title_bytecode_and_logs_once() {
        let dir = TempDir::new("gap");
        let original = b"synthetic console shader for the coverage check".to_vec();
        std::fs::write(dir.0.join(file_name(&original)), b"override bitcode").unwrap();
        let hash = content_hash(&original);
        std::fs::write(dir.0.join(format!("{hash:016x}.json")), r#"{"registers":[{"class":"srv","register":99}]}"#).unwrap();
        let l = layout(0);
        assert_eq!(lookup_checked_in(&dir.0, &original, Some(&l), 0), None, "register t99 is missing");
        assert!(!log_gap_once(hash, "again"), "already logged by the refusal above");
        std::fs::write(dir.0.join(format!("{hash:016x}.json")), r#"{"registers":[{"class":"srv","register":1}]}"#).unwrap();
        assert!(lookup_checked_in(&dir.0, &original, Some(&l), 0).is_some(), "now covered");
    }

    #[test]
    fn without_a_report_or_a_layout_the_override_passes() {
        let dir = TempDir::new("noreport");
        let original = b"synthetic console shader without a report".to_vec();
        std::fs::write(dir.0.join(file_name(&original)), b"override bitcode").unwrap();
        assert!(lookup_checked_in(&dir.0, &original, Some(&layout(0)), 0).is_some(), "no json beside it");
        let hash = content_hash(&original);
        std::fs::write(dir.0.join(format!("{hash:016x}.json")), r#"{"registers":[{"class":"srv","register":99}]}"#).unwrap();
        assert!(lookup_checked_in(&dir.0, &original, None, 0).is_some(), "root signature unknown");
        std::fs::write(dir.0.join(format!("{hash:016x}.json")), "not json").unwrap();
        assert!(lookup_checked_in(&dir.0, &original, Some(&layout(0)), 0).is_some(), "unreadable report");
        assert_eq!(lookup_checked_in(&dir.0, b"other bytes", Some(&layout(0)), 0), None, "a miss stays a miss");
    }

    #[test]
    fn the_report_ignores_unknown_fields_and_defaults_the_space() {
        let r = report(r#"{"registers":[{"class":"uav","register":7,"heap":false}],"extra":1}"#);
        assert_eq!(r.registers, vec![ReportRegister { class: ReportClass::Uav, register: 7, space: 0 }]);
        assert!(serde_json::from_str::<BindingReport>(r#"{"registers":[{"class":"bogus","register":1}]}"#).is_err());
    }

    #[test]
    fn root_layouts_are_remembered_by_object_address() {
        assert!(root_layout(0xDEAD_0001).is_none());
        remember_root_layout(0xDEAD_0001, layout(0x400));
        assert_eq!(root_layout(0xDEAD_0001).map(|l| l.flags), Some(0x400));
    }

    #[test]
    fn a_miss_logs_once_per_distinct_content_hash() {
        let original = b"a shader this test never saw an override for".to_vec();
        let hash = content_hash(&original);
        let first = log_miss_once(hash, original.len(), None, None);
        let second = log_miss_once(hash, original.len(), None, None);
        assert!(first, "the first report of a hash logs it");
        assert!(!second, "the same hash again does not log a second time");
        let other = b"a different shader, a different hash".to_vec();
        assert_ne!(hash, content_hash(&other));
        assert!(log_miss_once(content_hash(&other), other.len(), None, None), "a new hash still logs");
    }

    #[test]
    fn the_size_cap_defaults_to_four_mib_and_zero_means_no_cap() {
        assert_eq!(size_cap_from(None), 4 << 20);
        assert_eq!(size_cap_from(Some("not a number")), 4 << 20);
        assert_eq!(size_cap_from(Some("-1")), 4 << 20);
        assert_eq!(size_cap_from(Some("0")), u64::MAX);
        assert_eq!(size_cap_from(Some(" 1024 ")), 1024);
    }

    #[test]
    fn an_override_over_the_cap_is_a_miss_and_one_at_the_cap_is_a_hit() {
        let dir = TempDir::new("size-cap");
        let original = b"a shader whose override file is bigger than the cap in this test".to_vec();
        let data = vec![0x44u8; 100];
        std::fs::write(dir.0.join(file_name(&original)), &data).expect("the override file");
        assert!(find_capped(&dir.0, &original, 99).is_none(), "a 100 byte file over a 99 byte cap is not read");
        let (bytes, _) = find_capped(&dir.0, &original, 100).expect("a file exactly at the cap is read");
        assert_eq!(&bytes[bytes.len() - data.len()..], data.as_slice());
        assert!(find_capped(&dir.0, &original, u64::MAX).is_some(), "no cap reads any size");
    }
}
