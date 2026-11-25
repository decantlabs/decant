//! The pipeline manifest of the direct shader path (issue #172): one line per distinct pipeline
//! description the title creates, with everything the direct path needs to translate and compile it
//! later without the game. `decant_shader_warm` reads it and fills the DXIL cache ahead of time, and the
//! census of the translator (`--pipelines`) reads the stage hashes and the root signatures.
//!
//! `DECANT_D3D12X_PIPELINE_MANIFEST=<file>` turns it on. Records are appended at pipeline creation
//! (`lazy_pso::defer`), not at bind, because the title creates every pipeline at boot. The run that
//! writes a manifest must have the switches of the direct path on (`DECANT_D3D12X_DIRECT=1`): the root
//! signatures are recorded as the PC runtime got them, with the parameters and flags that path adds,
//! and they are only known to the layer while it is on.
//!
//! A manifest holds hashes and descriptions only, never shader bytes (they are title data: the
//! containers come from the title's shader archive by hash, or from the layer's dump of containers
//! that are in no archive). Like every file produced from a title it stays outside the repository.
//!
//! # Format, version 1
//!
//! UTF-8 text, one record per line, fields separated by one space. Lines starting with `#` are
//! comments, the first line is `# decant pipeline manifest 1`. Numbers are decimal unless they are
//! hashes (16 lower case hex digits) or flags (`0x` and hex digits).
//!
//! A root signature, written once before the first pipeline that names it:
//!
//! ```text
//! R <id> <version> <flags> <param>... <sampler>...
//! ```
//!
//! * `<id>`: FNV-1a 64 of the text after `R <id> `, so equal layouts have equal ids in every run.
//! * `<version>`: the RTS0 version word (1, 2, 3). `<flags>`: `D3D12_ROOT_SIGNATURE_FLAGS`.
//! * one token per root parameter, in root index order, `<kind>:<visibility>:<fields>`:
//!   * `table:<vis>:<range>+<range>...` (nothing after the second colon for a table without ranges),
//!     a range is `<class>.<base>.<space>.<count>.<offset>.<flags>` with the class `t`, `u`, `b`
//!     or `s` and 4294967295 for an unbounded count or an offset after one,
//!   * `const:<vis>:<register>.<space>.<count>` for root constants,
//!   * `cbv:<vis>:<register>.<space>`, `srv:...`, `uav:...` for root descriptors,
//! * then one token per static sampler, `sampler:<vis>:<register>.<space>`.
//!
//! The text is the parsed layout of the layer (`rts0::RootLayout`) without loss, with the
//! layer's own parameters (user data mirror, heap map, view shadow in space 15) as the last ones.
//! `RootLayout::without_layer_params` gives the title's own part.
//!
//! A pipeline:
//!
//! ```text
//! P <root id or -> topo=<n> msaa=<n> ds=<format> rt=<format>,... so=<0|1> in=<element>,... <TAG> <hash>...
//! ```
//!
//! * the root id names an `R` line, `-` when the layer did not know the root signature,
//! * the linkage (`shader_source::Linkage`): `D3D12_PRIMITIVE_TOPOLOGY_TYPE`, sample count, depth
//!   format, render target formats, stream output, and the input layout, an element being
//!   `<semantic>.<index>.<format>.<slot>.<offset>.<v|i>.<step rate>` (`i` per instance) with every
//!   byte of the semantic outside `A-Z a-z 0-9 _` written as `%XX`. A compute pipeline has zeros
//!   and empty lists there.
//! * the stages in pipeline order (VS, PS, DS, HS, GS, or CS, or AS, MS, PS), each a tag (the
//!   upper case short name, `VS PS DS HS GS CS AS MS`) and the FNV-1a 64 of the console container
//!   (`decant_shader::container::fnv1a64`, the name of the layer's override and dump files).
//!   The census reads such tag and hash pairs from any line.
//!
//! Every distinct `P` line is written once per process. A file that exists is appended to, so
//! readers drop repeated lines.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::direct_cache::Fnv128;
use crate::rts0::{RegisterClass, RootLayout, RootParam, RootParamKind, RootRange, StaticSampler};
use crate::shader_source::{InputElement, Linkage, StageKind};

/// The switch that names the manifest file.
pub const SWITCH: &str = "DECANT_D3D12X_PIPELINE_MANIFEST";
/// The first line of a manifest.
pub const HEADER: &str = "# decant pipeline manifest 1";

/// One pipeline of a manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// The id of its root signature, `None` when the layer did not know it.
    pub root: Option<u64>,
    /// How its stages link. `linkage.stages` are the kinds of `stages`.
    pub linkage: Linkage,
    /// Per stage the kind and the container hash.
    pub stages: Vec<(StageKind, u64)>,
}

/// A manifest read back.
#[derive(Clone, Debug, Default)]
pub struct Manifest {
    /// The root signatures by id.
    pub roots: HashMap<u64, RootLayout>,
    /// The distinct pipelines in the order they were first written.
    pub pipelines: Vec<Record>,
}

fn class_letter(c: RegisterClass) -> char {
    match c {
        RegisterClass::Srv => 't',
        RegisterClass::Uav => 'u',
        RegisterClass::Cbv => 'b',
        RegisterClass::Sampler => 's',
    }
}

fn class_of(letter: &str) -> Option<RegisterClass> {
    Some(match letter {
        "t" => RegisterClass::Srv,
        "u" => RegisterClass::Uav,
        "b" => RegisterClass::Cbv,
        "s" => RegisterClass::Sampler,
        _ => return None,
    })
}

/// The text of `layout` in an `R` line, without `R` and the id.
pub fn root_text(layout: &RootLayout) -> String {
    let mut out = format!("{} {:#x}", layout.version, layout.flags);
    for p in &layout.params {
        let v = p.visibility;
        let token = match &p.kind {
            RootParamKind::Table(ranges) => {
                let r: Vec<String> = ranges.iter().map(|r| format!("{}.{}.{}.{}.{}.{}", class_letter(r.class), r.base, r.space, r.count, r.offset, r.flags)).collect();
                format!("table:{v}:{}", r.join("+"))
            }
            RootParamKind::Constants { register, space, count } => format!("const:{v}:{register}.{space}.{count}"),
            RootParamKind::Descriptor { class, register, space } => {
                let kind = match class {
                    RegisterClass::Cbv => "cbv",
                    RegisterClass::Srv => "srv",
                    // A root descriptor is never a sampler, the parser of the layer refuses it.
                    RegisterClass::Uav | RegisterClass::Sampler => "uav",
                };
                format!("{kind}:{v}:{register}.{space}")
            }
        };
        out.push(' ');
        out.push_str(&token);
    }
    for s in &layout.static_samplers {
        out.push_str(&format!(" sampler:{}:{}.{}", s.visibility, s.register, s.space));
    }
    out
}

/// The id of a root signature text.
pub fn root_id(text: &str) -> u64 {
    decant_shader::container::fnv1a64(text.as_bytes())
}

fn numbers<const N: usize>(text: &str, what: &str) -> Result<[u32; N], String> {
    let parts: Vec<&str> = text.split('.').collect();
    if parts.len() != N {
        return Err(format!("{what} `{text}` needs {N} numbers"));
    }
    let mut out = [0u32; N];
    for (o, p) in out.iter_mut().zip(parts) {
        *o = p.parse().map_err(|_| format!("{what} `{text}`: `{p}` is no number"))?;
    }
    Ok(out)
}

/// The layout of an `R` line text (as [`root_text`] writes it).
pub fn parse_root(text: &str) -> Result<RootLayout, String> {
    let mut tokens = text.split(' ');
    let version = tokens.next().and_then(|t| t.parse().ok()).ok_or("root signature without a version")?;
    let flags = tokens.next().and_then(|t| u32::from_str_radix(t.strip_prefix("0x")?, 16).ok()).ok_or("root signature without flags")?;
    let mut layout = RootLayout { version, flags, params: Vec::new(), static_samplers: Vec::new() };
    for token in tokens {
        let mut f = token.splitn(3, ':');
        let (Some(kind), Some(vis), Some(fields)) = (f.next(), f.next(), f.next()) else {
            return Err(format!("root token `{token}` is not kind:visibility:fields"));
        };
        let visibility: u32 = vis.parse().map_err(|_| format!("root token `{token}`: bad visibility"))?;
        let kind = match kind {
            "table" => {
                let mut ranges = Vec::new();
                for r in fields.split('+').filter(|r| !r.is_empty()) {
                    let (letter, rest) = r.split_once('.').ok_or_else(|| format!("range `{r}` has no class"))?;
                    let class = class_of(letter).ok_or_else(|| format!("range `{r}`: unknown class"))?;
                    let [base, space, count, offset, flags] = numbers::<5>(rest, "range")?;
                    ranges.push(RootRange { class, count, base, space, flags, offset });
                }
                RootParamKind::Table(ranges)
            }
            "const" => {
                let [register, space, count] = numbers::<3>(fields, "root constants")?;
                RootParamKind::Constants { register, space, count }
            }
            "cbv" | "srv" | "uav" => {
                let [register, space] = numbers::<2>(fields, "root descriptor")?;
                let class = match kind {
                    "cbv" => RegisterClass::Cbv,
                    "srv" => RegisterClass::Srv,
                    _ => RegisterClass::Uav,
                };
                RootParamKind::Descriptor { class, register, space }
            }
            "sampler" => {
                let [register, space] = numbers::<2>(fields, "static sampler")?;
                layout.static_samplers.push(StaticSampler { register, space, visibility });
                continue;
            }
            other => return Err(format!("unknown root token kind `{other}`")),
        };
        if !layout.static_samplers.is_empty() {
            return Err(format!("root parameter `{token}` after a static sampler"));
        }
        layout.params.push(RootParam { kind, visibility });
    }
    Ok(layout)
}

fn escape(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || b == b'_' {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn unescape(text: &str) -> Result<String, String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = text.get(i + 1..i + 3).ok_or_else(|| format!("semantic `{text}`: cut escape"))?;
            out.push(u8::from_str_radix(hex, 16).map_err(|_| format!("semantic `{text}`: bad escape"))?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| format!("semantic `{text}` is not UTF-8"))
}

fn tag(kind: StageKind) -> String {
    kind.short().to_ascii_uppercase()
}

fn kind_of(tag: &str) -> Option<StageKind> {
    [StageKind::Vertex, StageKind::Pixel, StageKind::Domain, StageKind::Hull, StageKind::Geometry, StageKind::Compute, StageKind::Amplification, StageKind::Mesh]
        .into_iter()
        .find(|k| k.short().eq_ignore_ascii_case(tag))
}

/// The `P` line of a pipeline (without the line end). `hashes` are the container hashes of the
/// stages of `linkage.stages`, in that order.
pub fn pipeline_line(root: Option<u64>, linkage: &Linkage, hashes: &[u64]) -> String {
    let root = root.map_or_else(|| "-".to_string(), |r| format!("{r:016x}"));
    let targets: Vec<String> = linkage.targets.iter().map(u32::to_string).collect();
    let inputs: Vec<String> = linkage
        .inputs
        .iter()
        .map(|e| format!("{}.{}.{}.{}.{}.{}.{}", escape(&e.semantic), e.index, e.format, e.slot, e.offset, if e.per_instance { 'i' } else { 'v' }, e.step_rate))
        .collect();
    let mut line = format!(
        "P {root} topo={} msaa={} ds={} rt={} so={} in={}",
        linkage.topology_type,
        linkage.sample_count,
        linkage.depth_format,
        targets.join(","),
        u8::from(linkage.stream_output),
        inputs.join(",")
    );
    for (kind, hash) in linkage.stages.iter().zip(hashes) {
        line.push_str(&format!(" {} {hash:016x}", tag(*kind)));
    }
    line
}

fn parse_element(text: &str) -> Result<InputElement, String> {
    let parts: Vec<&str> = text.split('.').collect();
    let [semantic, index, format, slot, offset, class, step] = parts[..] else {
        return Err(format!("input element `{text}` needs 7 fields"));
    };
    let n = |s: &str| s.parse::<u32>().map_err(|_| format!("input element `{text}`: `{s}` is no number"));
    Ok(InputElement {
        semantic: unescape(semantic)?,
        index: n(index)?,
        format: n(format)?,
        slot: n(slot)?,
        offset: n(offset)?,
        per_instance: match class {
            "i" => true,
            "v" => false,
            _ => return Err(format!("input element `{text}`: class must be v or i")),
        },
        step_rate: n(step)?,
    })
}

/// The pipeline of a `P` line.
pub fn parse_pipeline(line: &str) -> Result<Record, String> {
    let mut tokens = line.split(' ').filter(|t| !t.is_empty());
    if tokens.next() != Some("P") {
        return Err("not a pipeline line".into());
    }
    let root = match tokens.next().ok_or("pipeline line without a root")? {
        "-" => None,
        r => Some(u64::from_str_radix(r, 16).map_err(|_| format!("bad root id `{r}`"))?),
    };
    let mut linkage = Linkage::default();
    let mut stages = Vec::new();
    let mut rest: Vec<&str> = Vec::new();
    for t in tokens {
        let Some((key, value)) = t.split_once('=') else {
            rest.push(t);
            continue;
        };
        let n = || value.parse::<u32>().map_err(|_| format!("`{t}`: not a number"));
        let list = || value.split(',').filter(|v| !v.is_empty());
        match key {
            "topo" => linkage.topology_type = n()?,
            "msaa" => linkage.sample_count = n()?,
            "ds" => linkage.depth_format = n()?,
            "rt" => linkage.targets = list().map(|v| v.parse::<u32>().map_err(|_| format!("`{t}`: not a number"))).collect::<Result<_, _>>()?,
            "so" => linkage.stream_output = n()? != 0,
            "in" => linkage.inputs = list().map(parse_element).collect::<Result<_, _>>()?,
            // Fields of a later version are skipped.
            _ => {}
        }
    }
    for pair in rest.chunks(2) {
        let [t, h] = pair else { return Err(format!("stage `{}` without a hash", pair[0])) };
        let kind = kind_of(t).ok_or_else(|| format!("unknown stage tag `{t}`"))?;
        let hash = u64::from_str_radix(h, 16).map_err(|_| format!("bad container hash `{h}`"))?;
        stages.push((kind, hash));
    }
    if stages.is_empty() {
        return Err("pipeline line without stages".into());
    }
    linkage.stages = stages.iter().map(|(k, _)| *k).collect();
    Ok(Record { root, linkage, stages })
}

/// A manifest from its text: the root signatures, and every distinct pipeline once, in the order
/// of its first line. The error names the line.
pub fn parse(text: &str) -> Result<Manifest, String> {
    let mut m = Manifest::default();
    let mut seen = HashSet::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim_end();
        let at = |e: String| format!("line {}: {e}", n + 1);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("R ") {
            let (id, body) = rest.split_once(' ').ok_or_else(|| at("root signature without a body".into()))?;
            let id = u64::from_str_radix(id, 16).map_err(|_| at(format!("bad root id `{id}`")))?;
            if root_id(body) != id {
                return Err(at(format!("root id {id:016x} is not the hash of its text")));
            }
            m.roots.insert(id, parse_root(body).map_err(at)?);
        } else if line.starts_with("P ") {
            if seen.insert(line.to_string()) {
                m.pipelines.push(parse_pipeline(line).map_err(at)?);
            }
        } else {
            return Err(at("neither a root signature nor a pipeline".into()));
        }
    }
    if let Some(p) = m.pipelines.iter().find(|p| p.root.is_some_and(|r| !m.roots.contains_key(&r))) {
        return Err(format!("pipeline {} names root signature {:016x}, which has no R line", pipeline_line(p.root, &p.linkage, &p.stages.iter().map(|s| s.1).collect::<Vec<_>>()), p.root.unwrap_or(0)));
    }
    Ok(m)
}

/// The manifest in `path`.
pub fn read(path: &Path) -> Result<Manifest, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Writes a manifest: each root signature and each distinct pipeline once.
pub struct Writer {
    file: std::fs::File,
    roots: HashSet<u64>,
    lines: HashSet<u128>,
    /// Container hashes by the address and length of a shared container (`lazy_pso` shares them
    /// by content and never frees them), so a container is hashed once.
    hashes: HashMap<(usize, usize), u64>,
    /// Pipelines written.
    pub pipelines: u64,
}

impl Writer {
    /// A writer that appends to `path`, with the header when the file is new or empty.
    pub fn open(path: &Path) -> std::io::Result<Writer> {
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        if file.metadata()?.len() == 0 {
            file.write_all(format!("{HEADER}\n").as_bytes())?;
        }
        Ok(Writer { file, roots: HashSet::new(), lines: HashSet::new(), hashes: HashMap::new(), pipelines: 0 })
    }

    /// Records one pipeline: its root signature (as the PC runtime got it, `None` when unknown),
    /// its linkage and the console bytes of the stages of `linkage.stages`. Each line is written
    /// with one call and no buffer, so a title that is stopped loses nothing that was recorded.
    /// Whether the pipeline was new.
    pub fn record(&mut self, root: Option<&RootLayout>, linkage: &Linkage, stages: &[&[u8]]) -> std::io::Result<bool> {
        let hashes: Vec<u64> = stages.iter().map(|b| *self.hashes.entry((b.as_ptr() as usize, b.len())).or_insert_with(|| decant_shader::container::fnv1a64(b))).collect();
        let (root_line, id) = match root {
            Some(layout) => {
                let text = root_text(layout);
                let id = root_id(&text);
                (Some(format!("R {id:016x} {text}\n")), Some(id))
            }
            None => (None, None),
        };
        let line = pipeline_line(id, linkage, &hashes);
        if !self.lines.insert(Fnv128::default().bytes(line.as_bytes()).finish()) {
            return Ok(false);
        }
        if let (Some(r), Some(id)) = (root_line, id)
            && self.roots.insert(id)
        {
            self.file.write_all(r.as_bytes())?;
        }
        self.file.write_all(format!("{line}\n").as_bytes())?;
        self.pipelines += 1;
        Ok(true)
    }
}

/// The file `DECANT_D3D12X_PIPELINE_MANIFEST` names, read once.
pub fn configured() -> Option<PathBuf> {
    static PATH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var_os(SWITCH).filter(|v| !v.is_empty() && v != "0").map(PathBuf::from)).clone()
}

/// Records the pipeline the title created with the copied description `owned`, when the manifest
/// is on. Called from `lazy_pso::defer` for graphics, compute and mesh pipelines.
#[cfg(windows)]
pub(crate) fn created(owned: &crate::lazy_pso::Owned) {
    use crate::log::xs_log;
    use std::sync::Mutex;
    static WRITER: std::sync::OnceLock<Option<Mutex<Writer>>> = std::sync::OnceLock::new();
    static FAILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let Some(path) = configured() else { return };
    let writer = WRITER.get_or_init(|| match Writer::open(&path) {
        Ok(w) => {
            xs_log!("pipeline manifest: recording every created pipeline to {}", path.display());
            Some(Mutex::new(w))
        }
        Err(e) => {
            xs_log!("pipeline manifest: {} not opened ({e}), nothing is recorded", path.display());
            None
        }
    });
    let Some(writer) = writer else { return };
    // A pipeline whose root signature the layer does not know is not recorded: the direct path
    // refuses it in the game too.
    let pipeline = match crate::direct_pso::pipeline_of(owned) {
        Ok(p) => p,
        Err(r) => {
            if FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                xs_log!("pipeline manifest: a pipeline is not recorded ({r})");
            }
            return;
        }
    };
    let stages: Vec<&[u8]> = pipeline.stages.iter().map(|s| &*s.bytes).collect();
    let mut w = writer.lock().unwrap_or_else(|e| e.into_inner());
    match w.record(Some(&pipeline.root), &pipeline.linkage, &stages) {
        Ok(true) if w.pipelines.is_power_of_two() => xs_log!("pipeline manifest: {} pipelines, {} root signatures recorded", w.pipelines, w.roots.len()),
        Ok(_) => {}
        Err(e) => {
            if FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                xs_log!("pipeline manifest: write to {} failed ({e})", path.display());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_cache::tests::TempDir;

    /// A PC root signature with a table of two ranges (one unbounded), root constants, a root
    /// descriptor, a static sampler and the layer's mirror.
    fn layout() -> RootLayout {
        RootLayout {
            version: 2,
            flags: 0xC01,
            params: vec![
                RootParam {
                    kind: RootParamKind::Table(vec![
                        RootRange { class: RegisterClass::Srv, count: 1, base: 2, space: 1, flags: 8, offset: 0 },
                        RootRange { class: RegisterClass::Cbv, count: u32::MAX, base: 0, space: 1, flags: 0, offset: 1 },
                    ]),
                    visibility: 0,
                },
                RootParam { kind: RootParamKind::Table(Vec::new()), visibility: 5 },
                RootParam { kind: RootParamKind::Constants { register: 3, space: 0, count: 4 }, visibility: 5 },
                RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Uav, register: 7, space: 2 }, visibility: 1 },
                RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Srv, register: 0, space: 3 }, visibility: 0 },
                RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 },
            ],
            static_samplers: vec![StaticSampler { register: 4, space: 0, visibility: 5 }],
        }
    }

    fn graphics() -> Linkage {
        Linkage {
            stages: vec![StageKind::Vertex, StageKind::Pixel],
            inputs: vec![
                InputElement { semantic: "POSITION".into(), index: 0, format: 6, slot: 0, offset: 0, per_instance: false, step_rate: 0 },
                InputElement { semantic: "odd name.%".into(), index: 2, format: 16, slot: 1, offset: 12, per_instance: true, step_rate: 1 },
            ],
            targets: vec![28, 10],
            depth_format: 40,
            topology_type: 3,
            sample_count: 1,
            stream_output: false,
        }
    }

    #[test]
    fn a_root_signature_reads_back_as_written() {
        let text = root_text(&layout());
        assert_eq!(
            text,
            "2 0xc01 table:0:t.2.1.1.0.8+b.0.1.4294967295.1.0 table:5: const:5:3.0.4 uav:1:7.2 srv:0:0.3 cbv:0:1.15 sampler:5:4.0"
        );
        assert_eq!(parse_root(&text), Ok(layout()));
        assert_eq!(root_id(&text), root_id(&root_text(&layout())), "the id is the same in every run");
        assert!(parse_root("2 0xc01 sampler:5:4.0 cbv:0:1.15").is_err(), "parameters come before samplers");
        assert!(parse_root("2 0xc01 table:0:x.1.1.1.1.1").is_err());
        assert!(parse_root("2 0xc01 const:0:1.2").is_err());
    }

    #[test]
    fn a_pipeline_reads_back_as_written_and_names_its_stages_for_the_census() {
        let line = pipeline_line(Some(0x0123_4567_89ab_cdef), &graphics(), &[0x28d8_87aa_04ff_a15a, 0xb40d_9ca2_4fe3_d23d]);
        assert_eq!(
            line,
            "P 0123456789abcdef topo=3 msaa=1 ds=40 rt=28,10 so=0 in=POSITION.0.6.0.0.v.0,odd%20name%2E%25.2.16.1.12.i.1 VS 28d887aa04ffa15a PS b40d9ca24fe3d23d"
        );
        let r = parse_pipeline(&line).expect("parses");
        assert_eq!(r, Record { root: Some(0x0123_4567_89ab_cdef), linkage: graphics(), stages: vec![(StageKind::Vertex, 0x28d8_87aa_04ff_a15a), (StageKind::Pixel, 0xb40d_9ca2_4fe3_d23d)] });
        // A compute pipeline without a known root signature.
        let compute = Linkage { stages: vec![StageKind::Compute], ..Linkage::default() };
        let line = pipeline_line(None, &compute, &[7]);
        assert_eq!(line, "P - topo=0 msaa=0 ds=0 rt= so=0 in= CS 0000000000000007");
        assert_eq!(parse_pipeline(&line).map(|r| (r.root, r.linkage)), Ok((None, compute)));
        assert!(parse_pipeline("P - topo=0 VS").is_err());
        assert!(parse_pipeline("P - topo=0 XS 0000000000000007").is_err());
    }

    #[test]
    fn the_writer_records_each_pipeline_and_root_signature_once_and_appends() {
        let tmp = TempDir::new("manifest");
        std::fs::create_dir_all(&tmp.0).expect("dir");
        let path = tmp.0.join("pipelines.txt");
        let (vs, ps, cs) = (b"vertex container".to_vec(), b"pixel container".to_vec(), b"compute container".to_vec());
        let compute = Linkage { stages: vec![StageKind::Compute], ..Linkage::default() };
        {
            let mut w = Writer::open(&path).expect("open");
            assert_eq!(w.record(Some(&layout()), &graphics(), &[&vs, &ps]).ok(), Some(true));
            assert_eq!(w.record(Some(&layout()), &graphics(), &[&vs, &ps]).ok(), Some(false), "the same description again");
            assert_eq!(w.record(Some(&layout()), &compute, &[&cs]).ok(), Some(true));
            assert_eq!(w.record(None, &compute, &[&cs]).ok(), Some(true), "another root signature is another pipeline");
            assert_eq!(w.pipelines, 3);
        }
        // A second process appends, its repeated lines are dropped by the reader.
        {
            let mut w = Writer::open(&path).expect("open again");
            assert_eq!(w.record(Some(&layout()), &compute, &[&cs]).ok(), Some(true));
        }
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.starts_with(&format!("{HEADER}\n")));
        assert_eq!(text.matches(HEADER).count(), 1);
        assert_eq!(text.lines().filter(|l| l.starts_with("R ")).count(), 2, "one root line per writer");
        let m = read(&path).expect("parses");
        assert_eq!(m.pipelines.len(), 3);
        let id = root_id(&root_text(&layout()));
        assert_eq!(m.roots.get(&id), Some(&layout()));
        let fnv = decant_shader::container::fnv1a64;
        assert_eq!(m.pipelines[0], Record { root: Some(id), linkage: graphics(), stages: vec![(StageKind::Vertex, fnv(&vs)), (StageKind::Pixel, fnv(&ps))] });
        assert_eq!(m.pipelines[2].root, None);
    }

    /// What `created` records for a pipeline the title made: the copy of its description, its
    /// root signature as the layer keeps it, and the hash of the console container.
    #[cfg(windows)]
    #[test]
    #[allow(unsafe_code)]
    fn a_copied_description_is_recorded_with_its_root_signature() {
        let tmp = TempDir::new("manifest-copy");
        std::fs::create_dir_all(&tmp.0).expect("dir");
        let root_object = 0x5EED_0001_usize;
        crate::dxil_override::remember_root_layout(root_object, layout());
        let cs = b"console compute container".to_vec();
        let mut desc = [0u8; crate::lazy_pso::COMPUTE_DESC_SIZE];
        desc[0..8].copy_from_slice(&(root_object as u64).to_le_bytes());
        desc[8..16].copy_from_slice(&(cs.as_ptr() as u64).to_le_bytes());
        desc[16..24].copy_from_slice(&(cs.len() as u64).to_le_bytes());
        // SAFETY: the description and its shader are live, the root signature is only looked up.
        let owned = unsafe { crate::lazy_pso::copy_compute(desc.as_ptr().cast()) };
        let p = crate::direct_pso::pipeline_of(&owned).expect("known root signature");
        let mut w = Writer::open(&tmp.0.join("m.txt")).expect("open");
        let stages: Vec<&[u8]> = p.stages.iter().map(|s| &*s.bytes).collect();
        assert_eq!(w.record(Some(&p.root), &p.linkage, &stages).ok(), Some(true));
        let m = read(&tmp.0.join("m.txt")).expect("parses");
        let r = &m.pipelines[0];
        assert_eq!(r.stages, [(StageKind::Compute, decant_shader::container::fnv1a64(&cs))]);
        assert_eq!(r.root.and_then(|id| m.roots.get(&id)), Some(&layout()));
    }

    #[test]
    fn a_damaged_manifest_is_refused_with_its_line() {
        let root = format!("R {:016x} {}", root_id("2 0x0"), "2 0x0");
        assert!(parse(&format!("{HEADER}\n{root}\nP {:016x} topo=0 msaa=0 ds=0 rt= so=0 in= CS 0000000000000001\n", root_id("2 0x0"))).is_ok());
        let err = parse(&format!("{HEADER}\nR 0000000000000001 2 0x0\n")).expect_err("wrong id");
        assert!(err.starts_with("line 2: "), "{err}");
        assert!(parse("P 0000000000000009 topo=0 CS 0000000000000001\n").expect_err("unknown root").contains("no R line"));
        assert!(parse("Q what\n").is_err());
    }
}
