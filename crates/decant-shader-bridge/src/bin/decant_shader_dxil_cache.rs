//! Builds the DXIL override files the d3d12-xs layer reads, from the SPIR-V the DecantShader player
//! cache holds: for every shader `<hash>.dxil` (the container) and `<hash>.json` (the registers the
//! shader needs), with `hash` the FNV-1a 64 of the console container, as 16 lower case hex digits.
//!
//! The descriptors are placed at the Direct3D registers (or heap indices) the FSSM binding table and
//! the `.src` file give (`decant_shader_identify`), the push words follow the `push A B C` line, and the
//! vertex and pixel interfaces carry the console semantic names. Every container is checked with
//! the validator of dxil.dll. A shader that cannot be carried is counted by the cause of the refusal
//! (the categories of `decant_dxil::spirv::ImportError`, plus `identify:*`, `locate:*`, `write:*`
//! and `validate:*`).
//!
//! Usage: `decant_shader_dxil_cache <input folder> <output folder> [--measure <stage number>]
//! [--stage vertex|pixel|compute] [--jobs N, at most 8] [--limit N] [--every N] [--max-bytes N]
//! [--space-hints <file> [--only-hinted]] [--table-heap] [--only-hashes <file>] [--refusals <file>] [--list-hashes <file>]`
//!
//! `--space-hints` reads the file the d3d12-xs layer writes (`DECANT_D3D12X_SPACE_HINTS`, one line
//! `<hash> <class letter> <register> <old space> <new space>` per register of a refused override).
//! For the hashes it lists, those registers go to the hinted space instead of the one the container
//! or the default gives. A register hinted into several spaces takes the most frequent one.
//! `--only-hinted` builds just the hashes of the file.
//!
//! `--table-heap` reads every descriptor of a console table through the user data mirror and the heap
//! (`HeapAt`), also when the binding table names a register for it. The title's root signatures are
//! not the ones its shaders were compiled with: one root signature serves many shaders and its table
//! names only some of their registers, which the console runtime never checks. Through the heap the
//! shader reads the descriptor at the table offset the console code reads, so the root signature
//! only needs the mirror and the heap flags. The class of the register stays: a `b` register is read as a
//! constant buffer, because the slot holds a constant buffer view. `--only-hashes` builds just the console container hashes
//! the file lists, one per line as 16 hex digits (the `hash=` of the layer's refusal lines).
//!
//! A sampler the console shader holds as an immediate (its first descriptor word is a constant of the
//! code, the console form of a static sampler) goes to a register of space 12 listed in
//! `immediate_samplers.txt` in the output folder, one per line in hex. When the probe wrote the four
//! words of the sampler (`ssharp <index> #w0 #w1 #w2 #w3` in the `.src`), the register is
//! `0x01000000 | low 24 bits of FNV-1a 32 of the words` and the line is `<register> <w0> <w1> <w2> <w3>`;
//! otherwise the register is the first word and the line holds it alone. The d3d12-xs layer gives
//! every root signature a static sampler for each line: from the four words (filter, LOD range, bias,
//! border, address modes, anisotropy, comparison), or from the first word alone (address modes,
//! anisotropy and comparison, a linear filter).
//!
//! `--refusals <file>` writes one line per module that was not carried: the console container hash
//! (16 hex digits, `-` when the container cannot be read), the pack number, the stage, the category
//! and the detail, separated by tabs. A hash the layer reports as missing can be looked up there.
//! `--list-hashes <file>` writes the hash, pack number, stage and name of every shader of the pack
//! (also the ones no measure folder holds) and builds nothing.
//!
//! The input is a flat translation cache (`<pack number>.spv` and `<pack number>.src`) or, with
//! `--measure`, a `decant_shader_batch` measure folder of one stage (0 pixel, 1 vertex, 5 compute) whose
//! modules are named `<stage>-<offset>/<local>.spv`. The shaders come from `DECANT_SHADERS_RPF`
//! and `DECANT_SHADERS_DIR`, as for the other tools. The output belongs to the game: keep it
//! locally, never in the repository.
//!
//! The `.json` has the shape the layer reads: `{"registers":[{"class":"srv","register":188,"space":0}],
//! "resource_heap":false,"sampler_heap":false,"heap_constant_buffers":0}`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use decant_shader_bridge::describe::{self, Rows};
use decant_shader_bridge::container_bindings::{self, Binding, Lookup};
use decant_shader_bridge::dxil_origin::{placement, root_descriptor};
use decant_shader_bridge::identify::{PushSource, Resource, push_registers};
use decant_shader_bridge::py::Failure;
use decant_shader_bridge::session::Session;
use decant_shader_bridge::shaders::Config;
use decant_shader_core::meta;
use decant_dxil::ir::{ResourceClass, ResourceKind};
use decant_dxil::spirv::{Bindings, IndexExpr, PushLayout, RegisterClass, Target, DEVICE_MEMORY_SPACE, USER_DATA_DWORDS, USER_DATA_REGISTER, USER_DATA_SPACE, UserData, HeapView, import_with};

const USAGE: &str = "usage: decant_shader_dxil_cache <input folder> <output folder> [--measure <stage number>] [--stage vertex|pixel|compute] [--jobs N] [--limit N] [--every N] [--max-bytes N] [--space-hints <file> [--only-hinted]] [--table-heap] [--only-hashes <file>] [--refusals <file>] [--list-hashes <file>]";

struct Args {
    input: PathBuf,
    output: PathBuf,
    measure: Option<u8>,
    stage: Option<String>,
    jobs: usize,
    limit: usize,
    every: usize,
    /// Modules larger than this are not imported (0: no limit, the default).
    max_bytes: u64,
    /// Only this pack number.
    only: Option<i64>,
    /// The hints file of the layer.
    space_hints: Option<PathBuf>,
    /// Build only the hashes the hints file lists.
    only_hinted: bool,
    /// Table descriptors through the heap even when they have a register.
    table_heap: bool,
    /// The file of console container hashes to build, all when unset.
    only_hashes: Option<PathBuf>,
    /// The file that gets one line per refused module.
    refusals: Option<PathBuf>,
    /// The file that gets the hash of every shader of the pack, nothing built.
    list_hashes: Option<PathBuf>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args { input: PathBuf::new(), output: PathBuf::new(), measure: None, stage: None, jobs: 8, limit: 0, every: 1, max_bytes: 0, only: None, space_hints: None, only_hinted: false, table_heap: false, only_hashes: None, refusals: None, list_hashes: None };
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        let number = |name: &str, v: String| v.parse::<usize>().map_err(|_| format!("{name}: not a number: {v}"));
        match arg.as_str() {
            "--measure" => a.measure = Some(number(arg, value(arg)?)? as u8),
            "--stage" => a.stage = Some(value(arg)?),
            "--jobs" => a.jobs = number(arg, value(arg)?)?.clamp(1, 8),
            "--limit" => a.limit = number(arg, value(arg)?)?,
            "--only" => a.only = Some(number(arg, value(arg)?)? as i64),
            "--space-hints" => a.space_hints = Some(value(arg)?.into()),
            "--only-hinted" => a.only_hinted = true,
            "--table-heap" => a.table_heap = true,
            "--only-hashes" => a.only_hashes = Some(value(arg)?.into()),
            "--refusals" => a.refusals = Some(value(arg)?.into()),
            "--list-hashes" => a.list_hashes = Some(value(arg)?.into()),
            "--max-bytes" => a.max_bytes = number(arg, value(arg)?)? as u64,
            "--every" => a.every = number(arg, value(arg)?)?.max(1),
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => positional.push(arg.clone()),
        }
    }
    if a.only_hinted && a.space_hints.is_none() {
        return Err("--only-hinted needs --space-hints".into());
    }
    let [input, output] = positional.as_slice() else { return Err("two folders are needed".into()) };
    a.input = input.into();
    a.output = output.into();
    Ok(a)
}

struct Entry {
    pack_id: i64,
    spv: PathBuf,
    src: PathBuf,
}

fn entries(args: &Args, config: &Config) -> Result<Vec<Entry>, String> {
    let mut out = Vec::new();
    match args.measure {
        None => {
            let rd = std::fs::read_dir(&args.input).map_err(|e| format!("{}: {e}", args.input.display()))?;
            for p in rd.filter_map(|e| e.ok()).map(|e| e.path()) {
                if p.extension().is_some_and(|x| x == "spv")
                    && let Some(n) = p.file_stem().and_then(|s| s.to_string_lossy().parse::<i64>().ok())
                {
                    out.push(Entry { pack_id: n, src: p.with_extension("src"), spv: p });
                }
            }
        }
        Some(stage) => {
            let pack = config.out().join("shaders_gpu.pack");
            let mut indices: Vec<usize> = Vec::new();
            decant_extract_shaders::pack::read(&pack, |index, _, s| {
                if s == stage {
                    indices.push(index);
                }
                false
            })
            .map_err(|e| format!("{}: {e}", pack.display()))?;
            let prefix = format!("{stage}-");
            let rd = std::fs::read_dir(&args.input).map_err(|e| format!("{}: {e}", args.input.display()))?;
            for dir in rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&prefix))) {
                let offset: usize = dir.file_name().and_then(|n| n.to_string_lossy().split('-').nth(1).and_then(|o| o.parse().ok())).unwrap_or(0);
                let Ok(files) = std::fs::read_dir(&dir) else { continue };
                for p in files.filter_map(|e| e.ok()).map(|e| e.path()) {
                    if p.extension().is_some_and(|x| x == "spv")
                        && let Some(local) = p.file_stem().and_then(|s| s.to_string_lossy().parse::<usize>().ok())
                        && let Some(&pack_id) = indices.get(offset + local)
                    {
                        out.push(Entry { pack_id: pack_id as i64, src: p.with_extension("src"), spv: p });
                    }
                }
            }
        }
    }
    out.sort_by_key(|e| e.pack_id);
    Ok(out)
}

/// The space the layer wants a register in: `(old, new)` by `(class letter, register)`.
type RegisterHints = HashMap<(char, u32), (u32, u32)>;

/// The old space and the count of hint lines by new space, of one register of one hash.
type SpaceVotes = (u32, BTreeMap<u32, usize>);

/// A register hinted into several spaces: hash, class letter, register, (space, lines) pairs.
type Conflict = (u64, char, u32, Vec<(u32, usize)>);

/// The hints of a file by console container hash, and the registers hinted into several spaces.
#[derive(Default, Debug, PartialEq, Eq)]
struct Hints {
    by_hash: HashMap<u64, RegisterHints>,
    /// `(hash, class letter, register, [(space, count)])`.
    conflicts: Vec<Conflict>,
}

/// Parses the hints file: `<hash, 16 hex digits> <class letter t u b s> <register> <old space> <new space>`
/// per line, blank lines and `#` lines ignored, a line that does not parse is skipped. A register
/// hinted into several spaces takes the one with most lines (the lower space on a tie).
fn parse_hints(text: &str) -> Hints {
    // (hash, class, register) to the old space and the count of lines by new space.
    let mut counts: BTreeMap<(u64, char, u32), SpaceVotes> = BTreeMap::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let [hash, class, register, old, new] = f[..] else { continue };
        let (Ok(hash), Ok(register), Ok(old), Ok(new)) = (u64::from_str_radix(hash, 16), register.parse::<u32>(), old.parse::<u32>(), new.parse::<u32>()) else { continue };
        let mut letters = class.chars();
        let (Some(class), None) = (letters.next(), letters.next()) else { continue };
        if !matches!(class, 't' | 'u' | 'b' | 's') {
            continue;
        }
        *counts.entry((hash, class, register)).or_insert_with(|| (old, BTreeMap::new())).1.entry(new).or_default() += 1;
    }
    let mut out = Hints::default();
    for ((hash, class, register), (old, spaces)) in counts {
        // Most lines first, the lower space on a tie.
        let Some(best) = spaces.iter().min_by_key(|(space, n)| (std::cmp::Reverse(**n), **space)).map(|(space, _)| *space) else { continue };
        if spaces.len() > 1 {
            out.conflicts.push((hash, class, register, spaces.iter().map(|(s, n)| (*s, *n)).collect()));
        }
        out.by_hash.entry(hash).or_default().insert((class, register), (old, best));
    }
    out
}

/// The hashes of an `--only-hashes` file: the first word of each line, 16 hex digits.
fn parse_hashes(text: &str) -> HashSet<u64> {
    text.lines().filter_map(|l| l.split_whitespace().next()).filter_map(|w| u64::from_str_radix(w, 16).ok()).collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The category of a module `--only-hinted` leaves out: not a refusal, not counted.
const NOT_HINTED: &str = "skipped:not-hinted";

/// A shader that was not carried: the category counts it, the detail is for the first examples.
type Refusal = (String, String);

fn refuse<T>(category: &str, detail: impl ToString) -> Result<T, Refusal> {
    Err((category.to_owned(), detail.to_string()))
}

/// The register space of the translator's own buffers: binding 53 is the page table (`t0`), 54, 55 and 56
/// are plain `uint` buffers (`u0`, `u1`, `u2`).
const SERVICE_SPACE: u32 = 13;

/// Where the translator's own buffers go. Every module declares them, only the ones it reads and writes count.
fn service_buffers(b: &mut Bindings, base: i64) {
    b.bind(0, (base + 53) as u32, 0, Target::Register { class: RegisterClass::Srv, space: SERVICE_SPACE, register: 0 });
    for (k, binding) in [54, 55, 56].into_iter().enumerate() {
        b.bind(0, (base + binding) as u32, 0, Target::Register { class: RegisterClass::Uav, space: SERVICE_SPACE, register: k as u32 });
    }
}

/// The register space of the samplers a console shader holds as immediates (`s<first word>`), which the
/// layer adds to the root signatures as static samplers (the contract of d3d12-xs `rts0`).
const IMMEDIATE_SAMPLER_SPACE: u32 = 12;

/// The file in the output folder that lists the immediate sampler words of the carried shaders.
const IMMEDIATE_SAMPLERS_FILE: &str = "immediate_samplers.txt";

/// The register of an immediate sampler whose origin text is `#<first word in hex>`. Only a word with
/// the address, anisotropy and comparison fields alone (below 0x10000) is taken: the others are not
/// sampler words the layer can rebuild.
fn immediate_sampler(origin: &str) -> Option<Target> {
    let word = u32::from_str_radix(origin.trim().strip_prefix('#')?, 16).ok()?;
    (word < 0x1_0000).then_some(Target::Register { class: RegisterClass::Sampler, space: IMMEDIATE_SAMPLER_SPACE, register: word })
}

/// The four words of immediate sampler `index`, from the probe's `ssharp <index> #w0 #w1 #w2 #w3` line
/// (written when every word is a constant). `None` without that line, or when the words are not a
/// sampler the layer can rebuild: a first word with more than the address, anisotropy and comparison
/// fields, mip filter 3 (not a filter), or a border color pointer.
fn full_sampler(lines: &[String], index: i64) -> Option<[u32; 4]> {
    let prefix = format!("ssharp {index} ");
    let rest = lines.iter().find_map(|l| l.strip_prefix(prefix.as_str()))?;
    let parsed: Vec<u32> = rest.split_whitespace().map(|w| w.strip_prefix('#').and_then(|h| u32::from_str_radix(h, 16).ok())).collect::<Option<_>>()?;
    let words: [u32; 4] = parsed.try_into().ok()?;
    (words[0] < 0x1_0000 && (words[2] >> 26) & 3 != 3 && words[3] & 0x3FFF_FFFF == 0).then_some(words)
}

/// The register of a full immediate sampler in space 12: FNV-1a 32 of its words, low 24 bits, above
/// the first-word registers (below 0x10000), so samplers that share a first word but not their filter,
/// LOD range, bias or border get registers of their own.
fn full_sampler_register(words: [u32; 4]) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for b in words.iter().flat_map(|w| w.to_le_bytes()) {
        h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
    0x0100_0000 | (h & 0x00FF_FFFF)
}

/// One line of `immediate_samplers.txt`: the register alone (a first word, which the layer rebuilds),
/// or the register and the four words of the sampler.
fn immediate_line(register: u32, words: Option<[u32; 4]>) -> String {
    match words {
        None => format!("{register:08x}"),
        Some(w) => format!("{register:08x} {:08x} {:08x} {:08x} {:08x}", w[0], w[1], w[2], w[3]),
    }
}

/// Reads a line of `immediate_samplers.txt` back.
fn parse_immediate_line(line: &str) -> Option<(u32, Option<[u32; 4]>)> {
    let hex: Vec<u32> = line.split_whitespace().map(|w| u32::from_str_radix(w, 16).ok()).collect::<Option<_>>()?;
    match hex[..] {
        [register] => Some((register, None)),
        [register, a, b, c, d] => Some((register, Some([a, b, c, d]))),
        _ => None,
    }
}

/// The immediate sampler words a carried shader's report names.
fn immediate_words(json: &str) -> Vec<u32> {
    let Ok(v) = serde_json::from_str::<Value>(json) else { return Vec::new() };
    v["registers"]
        .as_array()
        .map(|regs| {
            regs.iter()
                .filter(|r| r["class"] == "sampler" && r["space"] == IMMEDIATE_SAMPLER_SPACE)
                .filter_map(|r| r["register"].as_u64().map(|w| w as u32))
                .collect()
        })
        .unwrap_or_default()
}

/// The register space of the console's registers when the container does not say (it has no `PSV0`).
/// The title's root signatures put most tables and root descriptors in space 1, `DECANT_SHADER_DXIL_SPACE` sets another.
fn console_space() -> u32 {
    std::env::var("DECANT_SHADER_DXIL_SPACE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1)
}

/// What the container said about the registers of a module, for the summary.
#[derive(Default, Clone, Debug)]
struct Notes {
    /// The container has no `PSV0`: the default space was used.
    no_psv: bool,
    /// Registers `identify` chose that no container binding has, `t11`.
    off: Vec<String>,
    /// Registers that several spaces of the container have (the first was taken).
    ambiguous: Vec<String>,
    /// A register whose space differs from the default: `t11 space 0`.
    corrected: Vec<String>,
    /// A register the hints file moved: `t11 space 1 to 0`.
    hinted: Vec<String>,
    /// The immediate samplers placed by their four words: register in space 12 and words.
    immediate: Vec<(u32, [u32; 4])>,
}

fn register_target(register: &str, list: Option<&[Binding]>, what: &str, hinted: Option<&RegisterHints>, notes: &mut Notes) -> Result<Target, String> {
    let (letter, number) = register.split_at(1);
    let number: u32 = number.parse().map_err(|_| format!("register {register} is not a number"))?;
    let class = letter.chars().next().unwrap_or('?');
    let hint = hinted.and_then(|h| h.get(&(class, number))).copied();
    let found = match list {
        None => Some(console_space()),
        Some(list) => match container_bindings::lookup(list, class, number, what) {
            Lookup::One(space) => Some(space),
            Lookup::Ambiguous(spaces) => {
                notes.ambiguous.push(register.to_owned());
                Some(spaces[0])
            }
            Lookup::Missing => None,
        },
    };
    // The root signature of the title is the truth: the hinted space wins when the hint was made for
    // the space this tool would have used, or when the container lacks the register.
    let space = match (found, hint) {
        (Some(space), Some((old, new))) if old == space => {
            notes.hinted.push(format!("{register} space {space} to {new}"));
            new
        }
        (None, Some((_, new))) => {
            notes.hinted.push(format!("{register} to space {new}"));
            new
        }
        (Some(space), _) => space,
        (None, None) => {
            notes.off.push(register.to_owned());
            return Err(format!("register {register} is not in the container"));
        }
    };
    if space != console_space() {
        notes.corrected.push(format!("{register} space {space}"));
    }
    register_class(register, space).ok_or_else(|| format!("register {register} has no class"))
}

fn register_class(register: &str, space: u32) -> Option<Target> {
    let (letter, number) = register.split_at(1);
    let register: u32 = number.parse().ok()?;
    let class = match letter {
        "t" => RegisterClass::Srv,
        "u" => RegisterClass::Uav,
        "b" => RegisterClass::Cbv,
        "s" => RegisterClass::Sampler,
        _ => return None,
    };
    Some(Target::Register { class, space, register })
}

/// The origin text of the resource `kind index` in the `.src` lines.
fn origin_text<'a>(lines: &'a [String], kind: &str, index: i64) -> &'a str {
    let prefix = format!("{kind} {index} ");
    lines.iter().find_map(|l| l.strip_prefix(prefix.as_str())).unwrap_or("")
}

/// The placement of every identified resource, the push layout of the `push A B C` line, and why a
/// resource that stays unplaced has no heap index (by `(binding, element)`).
fn bindings_of(resources: &[Resource], lines: &[String], base: i64, list: Option<&[Binding]>, hinted: Option<&RegisterHints>, table_heap: bool, notes: &mut Notes) -> (Bindings, HashMap<(i64, u32), String>) {
    let mut b = Bindings::new();
    let mut reasons = HashMap::new();
    let element = |r: &Resource| r.rank.filter(|&k| k < 0xFFFF_FFFF).unwrap_or(0) as u32;
    let mut current = None;
    for r in resources {
        let Some(binding) = r.binding else { continue };
        let key = (base + binding) as u32;
        let sampler = r.kind == "sampler";
        // A register of the wrong class (a sampler at `t12`) is a mistake of the binding table: the mirror serves it.
        let class_ok = |reg: &str| reg.starts_with('s') == sampler;
        let target = match (&r.register, r.heap) {
            (Some(reg), _) if !class_ok(reg) => None,
            (Some(reg), _) => match register_target(reg, list, &r.kind, hinted, notes) {
                Ok(t) => Some(t),
                Err(why) => {
                    reasons.insert((base + binding, element(r)), why);
                    None
                }
            },
            (None, Some(h)) if r.origin == "tas-samplers" => Some(Target::SamplerHeap(h as u32)),
            (None, Some(h)) => Some(Target::Heap(h as u32)),
            (None, None) if sampler && r.origin == "immediat" => match full_sampler(lines, r.index) {
                Some(words) => {
                    let register = full_sampler_register(words);
                    notes.immediate.push((register, words));
                    Some(Target::Register { class: RegisterClass::Sampler, space: IMMEDIATE_SAMPLER_SPACE, register })
                }
                None => immediate_sampler(origin_text(lines, &r.kind, r.index)),
            },
            // A root descriptor: the buffer address is in the mirror, the shader reaches it through the window.
            _ if r.kind == "buffer" => root_descriptor(origin_text(lines, &r.kind, r.index)).map(Target::Address),
            _ => None,
        };
        // Through the user data mirror: a table the binding table does not name, a descriptor read at a
        // computed offset. A register placement keeps the constant index case, the mirror serves arrays
        // indexed at run time.
        // A `b` register of the binding table is a constant buffer view, in the heap as at the register.
        let constant = matches!(target, Some(Target::Register { class: RegisterClass::Cbv, .. }));
        let heap = |e: IndexExpr| match (sampler, constant) {
            (true, _) => Target::SamplerHeapAt(e),
            (false, true) => Target::ConstantHeapAt(e),
            (false, false) => Target::HeapAt(e),
        };
        let computed = match origin_text_placement(r, lines, resources, base, &mut current) {
            Ok(e) => Some(heap(e)),
            Err(why) => {
                reasons.entry((base + binding, element(r))).or_insert(why);
                None
            }
        };
        match (target, computed) {
            // The heap gives the descriptor at the table offset the console reads, whatever the root signature names.
            (Some(Target::Register { .. }), Some(a)) if table_heap => {
                b.bind(0, key, element(r), a);
            }
            (Some(t), alt) => {
                b.bind(0, key, element(r), t);
                if let Some(a) = alt {
                    b.bind_dynamic(0, key, element(r), a);
                }
            }
            (None, Some(a)) => {
                b.bind(0, key, element(r), a);
            }
            (None, None) => {}
        }
    }
    service_buffers(&mut b, base);
    if let Some(line) = lines.iter().find(|l| l.starts_with("push ")) {
        let words: Vec<u32> = line.split_whitespace().skip(1).filter_map(|w| w.parse().ok()).collect();
        if let [data_end, offsets_end, sizes_end] = words[..] {
            let buffers = (0..sizes_end.saturating_sub(offsets_end))
                .map(|k| resources.iter().find(|r| r.kind == "buffer" && r.rank == Some(i64::from(k))).and_then(|r| r.binding).map(|binding| (0, (base + binding) as u32, k)))
                .collect();
            let mut layout = PushLayout { data_end, offsets_end, sizes_end, buffers, group_id: Vec::new(), registers: Vec::new() };
            for (word, source) in push_registers(lines) {
                match source {
                    PushSource::Sgpr(n) => layout.registers.push((word, n)),
                    PushSource::Dispatch(axis) => layout.group_id.push((word, axis)),
                    PushSource::Raytrace(_) => {}
                }
            }
            b.push = Some(layout);
        }
    }
    (b, reasons)
}

/// The heap index of a resource through the mirror, from its origin text.
fn origin_text_placement(r: &Resource, lines: &[String], resources: &[Resource], base: i64, current: &mut Option<(u32, u32, u32)>) -> Result<IndexExpr, String> {
    if r.origin == "immediat" || r.origin == "tas" || r.origin == "tas-samplers" {
        return Err("not a table read".into());
    }
    placement(origin_text(lines, &r.kind, r.index), &r.kind, resources, base, current)
}

/// Names the interface of a located module with the console semantics.
fn name_elements(b: &mut Bindings, elements: &[Value], output: bool) {
    for e in elements {
        let (Some(semantic), Some(index)) = (e["semantic"].as_str(), e["index"].as_i64()) else { continue };
        if matches!(semantic, "SV_Position" | "SV_ClipDistance" | "SV_CullDistance") {
            if let Some(register) = e["register"].as_i64() {
                b.system_register(output, semantic, register as u32);
            }
        } else if !semantic.starts_with("SV_")
            && let Some(location) = e["location"].as_i64()
        {
            // Elements packed into one register differ by their first component.
            let component = e["mask"].as_u64().filter(|&m| m != 0).map_or(0, |m| m.trailing_zeros());
            b.name(output, location as u32, component, semantic, index as u32);
        }
    }
}

/// The `.json` the layer reads: the registers the shader needs and whether it indexes the heaps.
/// The user data dword and the table slot of a heap index that is a mirror dword plus constants (a
/// descriptor of a console table), `None` for any other index.
fn table_slot(e: &IndexExpr) -> Option<(u32, u32)> {
    fn parts(e: &IndexExpr) -> Option<(Option<u32>, u32)> {
        match e {
            IndexExpr::Const(c) => Some((None, *c)),
            IndexExpr::User(d) => Some((Some(*d), 0)),
            IndexExpr::Add(a, b) => {
                let ((da, ca), (db, cb)) = (parts(a)?, parts(b)?);
                Some((match (da, db) { (Some(_), Some(_)) => return None, (d, None) | (None, d) => d }, ca.checked_add(cb)?))
            }
            _ => None,
        }
    }
    parts(e).and_then(|(d, c)| d.map(|d| (d, c)))
}

/// The kind a heap view is read as, as the layer's `table_views` names it.
fn view_kind(k: ResourceKind) -> &'static str {
    match k {
        ResourceKind::TypedBuffer => "typed_buffer",
        ResourceKind::RawBuffer => "raw_buffer",
        ResourceKind::StructuredBuffer => "structured_buffer",
        ResourceKind::CBuffer => "cbuffer",
        ResourceKind::Sampler => "sampler",
        ResourceKind::Texture1D => "texture1d",
        ResourceKind::Texture2D => "texture2d",
        ResourceKind::Texture3D => "texture3d",
        ResourceKind::TextureCube => "texture_cube",
        ResourceKind::Texture1DArray => "texture1d_array",
        ResourceKind::Texture2DArray => "texture2d_array",
        ResourceKind::TextureCubeArray => "texture_cube_array",
    }
}

/// The `table_views` of the json: what the shader reads at each table offset (mirror dword and slot).
fn table_views(views: &[HeapView]) -> Vec<Value> {
    let mut seen = Vec::new();
    for v in views {
        let Some((dword, slot)) = table_slot(&v.index) else { continue };
        let class = match v.class {
            ResourceClass::Srv => "srv",
            ResourceClass::Uav => "uav",
            ResourceClass::CBuffer => "cbv",
            ResourceClass::Sampler => continue,
        };
        let one = json!({"dword": dword, "slot": slot, "class": class, "kind": view_kind(v.kind)});
        if !seen.contains(&one) {
            seen.push(one);
        }
    }
    seen
}

fn report_json(report: &decant_dxil::BindingReport, heap_constant_buffers: usize, user_data: Option<&UserData>, window: bool, heap_views: &[HeapView]) -> Value {
    let registers: Vec<Value> = report
        .bound
        .iter()
        .map(|r| {
            let class = match r.class {
                ResourceClass::Srv => "srv",
                ResourceClass::Uav => "uav",
                ResourceClass::CBuffer => "cbv",
                ResourceClass::Sampler => "sampler",
            };
            json!({"class": class, "register": r.register, "space": r.space})
        })
        .collect();
    // A report without the count is of a build that read every buffer of the heap raw.
    let mut v = json!({"registers": registers, "resource_heap": report.resource_heap, "sampler_heap": report.sampler_heap, "heap_constant_buffers": heap_constant_buffers});
    // The layer appends the mirror (a root constant buffer) only when the shader reads it.
    if let Some(u) = user_data {
        v["user_data"] = json!({"space": USER_DATA_SPACE, "register": USER_DATA_REGISTER, "dwords": USER_DATA_DWORDS, "used": u.dwords});
    }
    // The translator's own buffers (page table, touched and missing page bit sets, two more), see the README.
    let service: Vec<Value> = report
        .bound
        .iter()
        .filter(|r| r.space == SERVICE_SPACE)
        .map(|r| {
            let role = match (r.class, r.register) {
                (ResourceClass::Srv, 0) => "page_table",
                (ResourceClass::Uav, 0) => "pages_touched",
                (ResourceClass::Uav, 1) => "buffer_55",
                _ => "buffer_56",
            };
            json!({"role": role, "class": if r.class == ResourceClass::Srv { "srv" } else { "uav" }, "register": r.register, "space": r.space})
        })
        .collect();
    if !service.is_empty() {
        v["service"] = Value::Array(service);
    }
    // What the shader reads at each offset of its tables, for the layer's kind check (DECANT_D3D12X_KIND_CHECK).
    let views = table_views(heap_views);
    if !views.is_empty() {
        v["table_views"] = Value::Array(views);
    }
    if window {
        // Root UAV u0 (the window) and root CBV b0 (16 bytes, the base address, low dword first) of this space.
        v["window"] = json!({"space": DEVICE_MEMORY_SPACE, "uav_register": 0, "base_register": 0});
    }
    v
}

struct Built {
    hash: u64,
    dxil: Vec<u8>,
    json: String,
}

fn build(session: &Session, rows: &Rows, entry: &Entry, notes: &mut Notes, args: &Args, hints: Option<&Hints>, only_hashes: Option<&HashSet<u64>>) -> Result<(String, Built), Refusal> {
    let max_bytes = args.max_bytes;
    let failure = |f: Failure| -> Refusal { ("identify:error".into(), f.to_string()) };
    let pack_id = entry.pack_id;
    let meta_row = rows.meta.get(pack_id as usize).ok_or_else(|| ("identify:no-metadata".to_owned(), pack_id.to_string()))?;
    let stage = meta_row.get("stage").and_then(Value::as_str).unwrap_or("").to_owned();
    if !matches!(stage.as_str(), "vertex" | "pixel" | "compute") {
        return refuse(&format!("stage:{stage}"), "this stage is not translated");
    }
    let raw = std::fs::read(&entry.spv).map_err(|e| ("input:read".to_owned(), e.to_string()))?;
    if max_bytes > 0 && raw.len() as u64 > max_bytes {
        return refuse("input:too-large", format!("{} bytes of SPIR-V, the limit is {max_bytes}", raw.len()));
    }
    let text = std::fs::read(&entry.src).map_err(|e| ("input:src".to_owned(), e.to_string()))?;
    let lines: Vec<String> = String::from_utf8_lossy(&text).lines().map(str::to_owned).collect();
    let outputs = rows.outputs.get(pack_id as usize).and_then(Value::as_array).ok_or_else(|| ("identify:no-outputs".to_owned(), pack_id.to_string()))?;
    let located = stage != "compute";
    let spv = if located {
        let (inputs_map, outputs_map) = decant_shader_core::locate::tables(meta_row, outputs).map_err(|e| ("locate:tables".to_owned(), e))?;
        decant_shader_core::spirv::locate(&raw, &inputs_map, &outputs_map).map_err(|e| ("locate:module".to_owned(), e))?
    } else {
        raw
    };
    // The console container: its PSV0 gives the space of every register, and its hash names the files.
    let console = session.container(pack_id).map_err(failure)?;
    let hash = fnv1a64(&console);
    let hinted = hints.and_then(|h| h.by_hash.get(&hash));
    if (args.only_hinted && hinted.is_none()) || only_hashes.is_some_and(|h| !h.contains(&hash)) {
        return refuse(NOT_HINTED, "");
    }
    let resources = match session.identify(pack_id, &lines) {
        Ok(r) => r,
        Err(Failure::Exit(why)) => return refuse("identify:refused", why),
        Err(other) => return Err(failure(other)),
    };
    let base = if stage == "pixel" { 60 } else { 0 };
    let container_list = container_bindings::bindings(&console);
    notes.no_psv = container_list.is_none();
    let (mut bindings, reasons) = bindings_of(&resources, &lines, base, container_list.as_deref(), hinted, args.table_heap, notes);
    if located {
        let inputs = describe::inputs_of(meta::list(meta_row, "inputs").map_err(|e| ("identify:inputs".to_owned(), e))?).map_err(failure)?;
        name_elements(&mut bindings, &inputs, false);
        if stage == "vertex" {
            let outs = describe::outputs_of(outputs).map_err(failure)?;
            name_elements(&mut bindings, &outs, true);
        }
    }
    let words: Vec<u32> = spv.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    let imported = import_with(&words, &bindings).map_err(|e| {
        // An element the map lacks: say how the console shader chose it (`autre` is a run time choice).
        if e.category == "binding:unmapped" {
            let n: Vec<i64> = e.detail.split_whitespace().filter_map(|w| w.parse().ok()).collect();
            if let [_, binding, element] = n[..] {
                let found = resources.iter().find(|r| r.binding.map(|b| b + base) == Some(binding) && r.rank.filter(|&k| k < 0xFFFF_FFFF).unwrap_or(0) == element);
                let why = match found {
                    None => format!("no origin line, binding {}", binding - base),
                    Some(r) => match reasons.get(&(binding, element as u32)) {
                        Some(reason) if reason.starts_with("register ") => return ("binding:identify-off".to_owned(), format!("{reason}: {}", e.detail)),
                        Some(reason) => format!("origin {}: {}", r.origin, reason.chars().map(|c| if c.is_ascii_digit() { 'N' } else { c }).collect::<String>()),
                        None => format!("origin {}", r.origin),
                    },
                };
                return (format!("binding:unmapped ({why})"), e.detail);
            }
        }
        (e.category, e.detail)
    })?;
    let written = decant_dxil::write(&imported.shader).map_err(|e| ("write:error".to_owned(), e.to_string()))?;
    #[cfg(windows)]
    {
        let mut copy = written.container.clone();
        if let Err(e) = decant_dxil::validate::validate_in_place(&mut copy) {
            let text = e.to_string();
            if let Some(dir) = std::env::var_os("DECANT_SHADER_DXIL_CACHE_REJECTED") {
                let _ = std::fs::write(Path::new(&dir).join(format!("{pack_id}.dxbc")), &written.container);
            }
            let first = text.lines().next().unwrap_or("").chars().take(300).collect::<String>();
            // The message names the rule, the cause is counted without the numbers of the shader.
            let cause = first.split(" at ").next().unwrap_or(&first).chars().take(90).collect::<String>();
            return refuse(&format!("validate:{}", cause.split(": ").last().unwrap_or(&cause)), first);
        }
    }
    let heap_constant_buffers = imported.shader.resources.iter().filter(|r| r.class == ResourceClass::CBuffer && r.binding == decant_dxil::ir::Binding::Heap).count();
    let json = report_json(&written.report, heap_constant_buffers, imported.user_data.as_ref(), imported.device_memory.is_some(), &imported.heap_views);
    Ok((stage, Built { hash, dxil: written.container, json: serde_json::to_string_pretty(&json).expect("json") }))
}

#[derive(Default)]
struct Counts {
    modules: usize,
    carried: usize,
    /// Modules by what the container said: registers that are not in it, ambiguous ones, spaces that
    /// differ from the default, no `PSV0` at all.
    off: usize,
    ambiguous: usize,
    corrected: usize,
    hinted: usize,
    no_psv: usize,
    examples: Vec<String>,
    refused: BTreeMap<String, (usize, Vec<String>)>,
}

fn run(args: &Args) -> Result<(), String> {
    let hints = match &args.space_hints {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let hints = parse_hints(&text);
            let registers: usize = hints.by_hash.values().map(HashMap::len).sum();
            println!("hints: {} hashes, {registers} registers, {} registers hinted into several spaces", hints.by_hash.len(), hints.conflicts.len());
            for (hash, class, register, spaces) in hints.conflicts.iter().take(10) {
                println!("  conflict {hash:016x} {class}{register}: {spaces:?} (space, lines), the most frequent is taken");
            }
            Some(hints)
        }
        None => None,
    };
    let only_hashes = match &args.only_hashes {
        Some(path) => Some(parse_hashes(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?)),
        None => None,
    };
    let config = Config::from_env();
    if let Some(path) = &args.list_hashes {
        let rows = Rows::load(&config).map_err(|e| e.to_string())?;
        let session = Session::open(config).map_err(|e| e.to_string())?;
        let mut text = String::new();
        for (pack_id, meta_row) in rows.meta.iter().enumerate() {
            let hash = session.container(pack_id as i64).map_or_else(|_| "-".to_owned(), |c| format!("{:016x}", fnv1a64(&c)));
            let field = |k: &str| meta_row.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
            text.push_str(&format!("{hash}\t{pack_id}\t{}\t{}\n", field("stage"), field("name")));
        }
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("{} shaders listed in {}", rows.meta.len(), path.display());
        return Ok(());
    }
    let all = entries(args, &config)?;
    let rows = Rows::load(&config).map_err(|e| e.to_string())?;
    let session = Session::open(config).map_err(|e| e.to_string())?;
    let mut list: Vec<&Entry> = all.iter().step_by(args.every).filter(|e| args.only.is_none_or(|o| o == e.pack_id)).collect();
    if args.limit > 0 {
        list.truncate(args.limit);
    }
    std::fs::create_dir_all(&args.output).map_err(|e| format!("{}: {e}", args.output.display()))?;
    let next = AtomicUsize::new(0);
    let counts: Mutex<BTreeMap<String, Counts>> = Mutex::new(BTreeMap::new());
    let refusal_lines: Mutex<Vec<String>> = Mutex::new(Vec::new());
    // Register in space 12 -> the four words when the probe gave them (`None`: a first word alone).
    let words: Mutex<BTreeMap<u32, Option<[u32; 4]>>> = Mutex::new(BTreeMap::new());
    let clashes: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..args.jobs {
            std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn_scoped(scope, || {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(entry) = list.get(i) else { break };
                        let stage_of = |rows: &Rows| rows.meta.get(entry.pack_id as usize).and_then(|m| m.get("stage")).and_then(Value::as_str).unwrap_or("?").to_owned();
                        let wanted = args.stage.as_deref();
                        if wanted.is_some_and(|w| w != stage_of(&rows)) {
                            continue;
                        }
                        if std::env::var_os("DECANT_SHADER_DXIL_CACHE_TRACE").is_some() {
                            eprintln!("module {}", entry.pack_id);
                        }
                        let mut notes = Notes::default();
                        let result = build(&session, &rows, entry, &mut notes, args, hints.as_ref(), only_hashes.as_ref());
                        if result.as_ref().is_err_and(|(category, _)| category == NOT_HINTED) {
                            continue;
                        }
                        let mut guard = counts.lock().expect("lock");
                        {
                            let c = guard.entry(stage_of(&rows)).or_default();
                            c.off += usize::from(!notes.off.is_empty());
                            c.ambiguous += usize::from(!notes.ambiguous.is_empty());
                            c.corrected += usize::from(!notes.corrected.is_empty());
                            c.hinted += usize::from(!notes.hinted.is_empty());
                            c.no_psv += usize::from(notes.no_psv);
                            if c.examples.len() < 2
                                && let Some(first) = notes.corrected.first()
                            {
                                c.examples.push(format!("{}: {first}", entry.pack_id));
                            }
                        }
                        match result {
                            Ok((stage, built)) => {
                                let c = guard.entry(stage).or_default();
                                c.modules += 1;
                                c.carried += 1;
                                let stem = args.output.join(format!("{:016x}", built.hash));
                                let _ = std::fs::write(stem.with_extension("dxil"), &built.dxil);
                                let _ = std::fs::write(stem.with_extension("json"), &built.json);
                                let mut known = words.lock().expect("lock");
                                for register in immediate_words(&built.json) {
                                    let full = notes.immediate.iter().find(|(r, _)| *r == register).map(|&(_, w)| w);
                                    match known.insert(register, full) {
                                        Some(old) if old != full => clashes.lock().expect("lock").push(format!("{register:08x}: {old:x?} and {full:x?}")),
                                        _ => {}
                                    }
                                }
                            }
                            Err((category, detail)) => {
                                if args.refusals.is_some() {
                                    let hash = session.container(entry.pack_id).map_or_else(|_| "-".to_owned(), |c| format!("{:016x}", fnv1a64(&c)));
                                    let detail: String = detail.chars().take(300).map(|c| if c == '\t' || c == '\n' { ' ' } else { c }).collect();
                                    refusal_lines.lock().expect("lock").push(format!("{hash}\t{}\t{}\t{category}\t{detail}", entry.pack_id, stage_of(&rows)));
                                }
                                let c = guard.entry(stage_of(&rows)).or_default();
                                c.modules += 1;
                                let r = c.refused.entry(category).or_default();
                                r.0 += 1;
                                if r.1.len() < 2 {
                                    r.1.push(format!("{}: {}", entry.pack_id, detail.chars().take(160).collect::<String>()));
                                }
                            }
                        }
                    }
                })
                .expect("thread");
        }
    });
    let mut words = words.into_inner().expect("lock");
    let clashes = clashes.into_inner().expect("lock");
    if !clashes.is_empty() {
        return Err(format!("immediate samplers: one register for two different samplers: {}", clashes.join("; ")));
    }
    if !words.is_empty() {
        let path = args.output.join(IMMEDIATE_SAMPLERS_FILE);
        let old = std::fs::read_to_string(&path).unwrap_or_default();
        for (register, full) in old.lines().filter_map(parse_immediate_line) {
            words.entry(register).or_insert(full);
        }
        let text: String = words.iter().map(|(&r, &w)| immediate_line(r, w) + "\n").collect();
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        let full = words.values().filter(|w| w.is_some()).count();
        println!("immediate samplers: {} in {} ({full} with their four words)", words.len(), path.display());
    }
    if let Some(path) = &args.refusals {
        let mut lines = refusal_lines.into_inner().expect("lock");
        lines.sort();
        let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let counts = counts.into_inner().expect("lock");
    for (stage, c) in &counts {
        println!("{stage}: {} modules, {} carried, {} refused", c.modules, c.carried, c.modules - c.carried);
        println!("  container: {} modules with a register identify chose that it lacks, {} ambiguous, {} with a space other than {}, {} without PSV0, {} moved by hints", c.off, c.ambiguous, c.corrected, console_space(), c.no_psv, c.hinted);
        for e in &c.examples {
            println!("    corrected e.g. {e}");
        }
        let mut by_count: Vec<_> = c.refused.iter().collect();
        by_count.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for (category, (n, examples)) in by_count {
            println!("  {n:6}  {category}");
            for e in examples {
                println!("          e.g. {e}");
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn an_immediate_sampler_goes_to_its_word_in_space_12() {
        assert_eq!(super::immediate_sampler("#92"), Some(Target::Register { class: RegisterClass::Sampler, space: 12, register: 0x92 }));
        assert_eq!(super::immediate_sampler("#3092"), Some(Target::Register { class: RegisterClass::Sampler, space: 12, register: 0x3092 }));
        assert_eq!(super::immediate_sampler("#3f800000"), None, "not a sampler word");
        assert_eq!(super::immediate_sampler("ReadConst(x)"), None);
        let json = r#"{"registers":[{"class":"sampler","register":146,"space":12},{"class":"sampler","register":3,"space":1},{"class":"srv","register":5,"space":12}]}"#;
        assert_eq!(super::immediate_words(json), vec![146]);
    }

    #[test]
    fn two_immediate_samplers_with_one_first_word_get_their_own_registers() {
        // The same address modes (0x92), one linear and one point filter (word 2): the case word 0 alone merged.
        let lines: Vec<String> = ["sampler 0 #92", "sampler 1 #92", "ssharp 0 #92 #c00000 #18500000 #0", "ssharp 1 #92 #c00000 #14000000 #0"].iter().map(|s| s.to_string()).collect();
        let mut first = resource("sampler", None, None, 3, 0, "immediat");
        let mut second = resource("sampler", None, None, 4, 0, "immediat");
        (first.index, second.index) = (0, 1);
        let mut notes = Notes::default();
        let (b, _) = bindings_of(&[first, second], &lines, 0, None, None, false, &mut notes);
        let (linear, point) = (super::full_sampler_register([0x92, 0xc0_0000, 0x1850_0000, 0]), super::full_sampler_register([0x92, 0xc0_0000, 0x1400_0000, 0]));
        assert_ne!(linear, point);
        assert!(linear >= 0x0100_0000 && point >= 0x0100_0000, "above the first-word registers");
        assert_eq!(b.target(0, 3, 0), Some(Target::Register { class: RegisterClass::Sampler, space: 12, register: linear }));
        assert_eq!(b.target(0, 4, 0), Some(Target::Register { class: RegisterClass::Sampler, space: 12, register: point }));
        assert_eq!(notes.immediate, vec![(linear, [0x92, 0xc0_0000, 0x1850_0000, 0]), (point, [0x92, 0xc0_0000, 0x1400_0000, 0])]);
        // Without the probe's `ssharp` line the first word is the register, as before.
        let mut notes = Notes::default();
        let (b, _) = bindings_of(&[resource("sampler", None, None, 3, 0, "immediat")], &["sampler 0 #92".to_owned()], 0, None, None, false, &mut notes);
        assert_eq!(b.target(0, 3, 0), Some(Target::Register { class: RegisterClass::Sampler, space: 12, register: 0x92 }));
        assert!(notes.immediate.is_empty());
    }

    #[test]
    fn words_that_are_not_a_sampler_keep_the_first_word_form() {
        let line = |s: &str| vec![s.to_owned()];
        assert_eq!(super::full_sampler(&line("ssharp 0 #92 #c00000 #18500000 #80000000"), 0), Some([0x92, 0xc0_0000, 0x1850_0000, 0x8000_0000]), "an opaque white border");
        assert_eq!(super::full_sampler(&line("ssharp 0 #3f800000 #0 #0 #0"), 0), None, "a float, not a first word");
        assert_eq!(super::full_sampler(&line("ssharp 0 #0 #0 #3f800000 #0"), 0), None, "mip filter 3");
        assert_eq!(super::full_sampler(&line("ssharp 0 #0 #0 #0 #3f800000"), 0), None, "a border color pointer");
        assert_eq!(super::full_sampler(&line("ssharp 0 #0 #0 #0"), 0), None, "three words");
        assert_eq!(super::full_sampler(&line("ssharp 1 #0 #0 #0 #0"), 0), None, "another sampler");
    }

    #[test]
    fn immediate_sampler_lines_read_back() {
        for (register, words) in [(0x92, None), (0x0123_4567, Some([0x92, 0xc0_0000, 0x1400_0000, 0]))] {
            assert_eq!(super::parse_immediate_line(&super::immediate_line(register, words)), Some((register, words)));
        }
        assert_eq!(super::immediate_line(0x92, None), "00000092");
        assert_eq!(super::parse_immediate_line("00000092 1 2"), None);
        assert_eq!(super::parse_immediate_line("zz"), None);
    }

    use super::*;
    use decant_dxil::ir::ResourceKind;
    use decant_dxil::{BindingReport, BoundResource};

    fn resource(kind: &str, register: Option<&str>, heap: Option<u64>, binding: i64, rank: i64, origin: &'static str) -> Resource {
        Resource { kind: kind.into(), index: 0, register: register.map(str::to_owned), origin, heap, binding: Some(binding), rank: Some(rank), table: None, offset: None }
    }

    #[test]
    fn registers_and_heap_indices_become_targets() {
        assert_eq!(register_class("t188", 1), Some(Target::Register { class: RegisterClass::Srv, space: 1, register: 188 }));
        assert_eq!(register_class("b27", 1), Some(Target::Register { class: RegisterClass::Cbv, space: 1, register: 27 }));
        assert_eq!(register_class("u9", 1), Some(Target::Register { class: RegisterClass::Uav, space: 1, register: 9 }));
        assert_eq!(register_class("s3", 1), Some(Target::Register { class: RegisterClass::Sampler, space: 1, register: 3 }));
        assert_eq!(register_class("x3", 1), None);
        let resources = [
            resource("buffer", Some("t188"), None, 0, 2, "table"),
            resource("image", None, Some(12), 3, 0, "tas"),
            resource("sampler", None, Some(4), 51, 0, "tas-samplers"),
            resource("buffer", None, None, 0, 5, "autre"),
        ];
        let (b, _) = bindings_of(&resources, &["push 0 1 3".to_owned()], 60, None, None, false, &mut Notes::default());
        assert_eq!(b.target(0, 60, 2), Some(Target::Register { class: RegisterClass::Srv, space: 1, register: 188 }));
        assert_eq!(b.target(0, 63, 0), Some(Target::Heap(12)));
        assert_eq!(b.target(0, 111, 0), Some(Target::SamplerHeap(4)));
        assert_eq!(b.target(0, 60, 5), None);
        let push = b.push.expect("push");
        assert_eq!((push.data_end, push.offsets_end, push.sizes_end), (0, 1, 3));
        assert_eq!(push.buffers, vec![None, None]);
    }

    #[test]
    fn the_container_decides_the_space_and_a_missing_register_is_off() {
        let list = [Binding { class: 'b', space: 1, lower: 27, upper: 27, kind: 13 }, Binding { class: 't', space: 0, lower: 11, upper: 11, kind: 12 }, Binding { class: 't', space: 3, lower: 11, upper: 11, kind: 2 }];
        let mut notes = Notes::default();
        assert_eq!(register_target("t11", Some(&list), "buffer", None, &mut notes), Ok(Target::Register { class: RegisterClass::Srv, space: 0, register: 11 }));
        assert_eq!(register_target("t11", Some(&list), "image", None, &mut notes), Ok(Target::Register { class: RegisterClass::Srv, space: 3, register: 11 }));
        assert_eq!(register_target("b27", Some(&list), "buffer", None, &mut notes), Ok(Target::Register { class: RegisterClass::Cbv, space: 1, register: 27 }));
        assert!(notes.corrected.contains(&"t11 space 0".to_owned()));
        assert!(notes.off.is_empty());
        let err = register_target("u9", Some(&list), "buffer", None, &mut notes).unwrap_err();
        assert!(err.contains("not in the container"), "{err}");
        assert_eq!(notes.off, vec!["u9".to_owned()]);
    }

    #[test]
    fn the_json_has_the_shape_the_layer_reads() {
        let report = BindingReport {
            bound: vec![
                BoundResource { class: ResourceClass::Srv, kind: ResourceKind::RawBuffer, space: 1, register: 188, range_size: 1, size_bytes: 0 },
                BoundResource { class: ResourceClass::CBuffer, kind: ResourceKind::CBuffer, space: 1, register: 27, range_size: 1, size_bytes: 64 },
            ],
            resource_heap: true,
            sampler_heap: false,
            wave_size: None,
        };
        let want = json!({"registers": [{"class": "srv", "register": 188, "space": 1}, {"class": "cbv", "register": 27, "space": 1}], "resource_heap": true, "sampler_heap": false, "heap_constant_buffers": 1});
        assert_eq!(report_json(&report, 1, None, false, &[]), want);
        let views = [
            HeapView { class: ResourceClass::Srv, kind: ResourceKind::RawBuffer, index: IndexExpr::User(9).plus(IndexExpr::Const(5)) },
            HeapView { class: ResourceClass::Uav, kind: ResourceKind::Texture2D, index: IndexExpr::Const(3).plus(IndexExpr::User(8)) },
            HeapView { class: ResourceClass::Srv, kind: ResourceKind::RawBuffer, index: IndexExpr::User(9).plus(IndexExpr::Const(5)) },
            HeapView { class: ResourceClass::Srv, kind: ResourceKind::RawBuffer, index: IndexExpr::Const(7) },
        ];
        let with = report_json(&report, 1, None, false, &views);
        assert_eq!(with["table_views"], json!([
            {"dword": 9, "slot": 5, "class": "srv", "kind": "raw_buffer"},
            {"dword": 8, "slot": 3, "class": "uav", "kind": "texture2d"},
        ]));
    }

    #[test]
    fn hints_are_parsed_and_the_most_frequent_space_wins() {
        let text = "# a comment\n\
            00000000000000ab t 11 1 0\n\
            00000000000000ab b 27 1 2\n\
            00000000000000ab b 27 1 3\n\
            00000000000000ab b 27 1 3\n\
            00000000000000cd u 4 1 0\n\
            00000000000000cd u 5 1 0\n\
            00000000000000cd u 5 1 2\n\
            not a hint\n\
            00000000000000ef x 1 1 0\n\
            00000000000000ef t 1 1\n";
        let hints = parse_hints(text);
        assert_eq!(hints.by_hash.len(), 2, "the broken lines are skipped");
        let ab = &hints.by_hash[&0xab];
        assert_eq!(ab[&('t', 11)], (1, 0));
        assert_eq!(ab[&('b', 27)], (1, 3), "two lines for space 3, one for 2");
        assert_eq!(hints.by_hash[&0xcd][&('u', 5)], (1, 0), "a tie takes the lower space");
        assert_eq!(hints.conflicts, vec![(0xab, 'b', 27, vec![(2, 1), (3, 2)]), (0xcd, 'u', 5, vec![(0, 1), (2, 1)])]);
    }

    #[test]
    fn a_hint_places_the_register_in_the_hinted_space() {
        let list = [Binding { class: 'b', space: 1, lower: 27, upper: 27, kind: 13 }, Binding { class: 't', space: 1, lower: 11, upper: 11, kind: 12 }];
        let mut hinted = RegisterHints::new();
        hinted.insert(('b', 27), (1, 4));
        hinted.insert(('t', 11), (7, 0));
        hinted.insert(('u', 9), (1, 2));
        let mut notes = Notes::default();
        assert_eq!(register_target("b27", Some(&list), "buffer", Some(&hinted), &mut notes), Ok(Target::Register { class: RegisterClass::Cbv, space: 4, register: 27 }));
        assert_eq!(notes.hinted, vec!["b27 space 1 to 4".to_owned()]);
        // a hint made for another space than the tool would use is stale and ignored
        assert_eq!(register_target("t11", Some(&list), "buffer", Some(&hinted), &mut notes), Ok(Target::Register { class: RegisterClass::Srv, space: 1, register: 11 }));
        // a register the container lacks is placed by the hint
        assert_eq!(register_target("u9", Some(&list), "buffer", Some(&hinted), &mut notes), Ok(Target::Register { class: RegisterClass::Uav, space: 2, register: 9 }));
        assert!(notes.off.is_empty());
        // no container: the default space is what the hint was made for
        assert_eq!(register_target("b27", None, "buffer", Some(&hinted), &mut notes), Ok(Target::Register { class: RegisterClass::Cbv, space: 4, register: 27 }));
        // without a hint nothing changes
        assert_eq!(register_target("b27", Some(&list), "buffer", None, &mut notes), Ok(Target::Register { class: RegisterClass::Cbv, space: 1, register: 27 }));
    }

    #[test]
    fn hinted_registers_reach_the_bindings() {
        let resources = [resource("buffer", Some("t188"), None, 0, 2, "table")];
        let mut hinted = RegisterHints::new();
        hinted.insert(('t', 188), (1, 0));
        let (b, _) = bindings_of(&resources, &[], 0, None, Some(&hinted), false, &mut Notes::default());
        assert_eq!(b.target(0, 0, 2), Some(Target::Register { class: RegisterClass::Srv, space: 0, register: 188 }));
    }

    #[test]
    fn table_heap_reads_a_table_register_through_the_heap() {
        let resources = [resource("buffer", Some("t188"), None, 0, 0, "table")];
        let lines = ["buffer 0 LoadAddressU32@352(GetAddressResource(GetUserData(s9),#0,),#0,#0,imm,)".to_owned()];
        let (b, _) = bindings_of(&resources, &lines, 0, None, None, false, &mut Notes::default());
        assert_eq!(b.target(0, 0, 0), Some(Target::Register { class: RegisterClass::Srv, space: 1, register: 188 }));
        let (b, _) = bindings_of(&resources, &lines, 0, None, None, true, &mut Notes::default());
        assert_eq!(b.target(0, 0, 0), Some(Target::HeapAt(IndexExpr::User(9).plus(IndexExpr::Const(11)))));
        // a register without a table origin stays a register
        let (b, _) = bindings_of(&resources, &[], 0, None, None, true, &mut Notes::default());
        assert_eq!(b.target(0, 0, 0), Some(Target::Register { class: RegisterClass::Srv, space: 1, register: 188 }));
    }

    #[test]
    fn table_heap_keeps_the_class_of_a_constant_buffer_register() {
        // The table of the dispatch the GPU was lost in: buffers at 0 to 2, the constant buffer at 3.
        let at = |n: u32| IndexExpr::User(5).plus(IndexExpr::Const(n));
        let mut resources = [resource("buffer", Some("u23"), None, 0, 0, "table"), resource("buffer", Some("b23"), None, 0, 1, "table")];
        resources[1].index = 1;
        let lines = [
            "buffer 0 LoadAddressU32@0(GetAddressResource(GetUserData(s5),#0,),#0,#0,imm,)".to_owned(),
            "buffer 1 LoadAddressU32@96(GetAddressResource(GetUserData(s5),#0,),#0,#0,imm,)".to_owned(),
        ];
        let (b, _) = bindings_of(&resources, &lines, 0, None, None, true, &mut Notes::default());
        assert_eq!(b.target(0, 0, 0), Some(Target::HeapAt(at(0))));
        assert_eq!(b.target(0, 0, 1), Some(Target::ConstantHeapAt(at(3))));
        // a descriptor the binding table does not name has no class
        resources[1].register = None;
        let (b, _) = bindings_of(&resources, &lines, 0, None, None, true, &mut Notes::default());
        assert_eq!(b.target(0, 0, 1), Some(Target::HeapAt(at(3))));
    }

    #[test]
    fn the_hash_file_is_one_hash_per_line() {
        let h = parse_hashes("92c2ef42c2d86dab

00000000000000ab extra words
not a hash
");
        assert_eq!(h, HashSet::from([0x92c2_ef42_c2d8_6dab, 0xab]));
    }

    #[test]
    fn arguments() {
        let list = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let a = parse(&list(&["in", "out", "--measure", "1", "--every", "4"])).unwrap();
        assert_eq!((a.measure, a.every, a.jobs), (Some(1), 4, 8));
        assert!(parse(&list(&["in"])).is_err());
        assert!(parse(&list(&["in", "out", "--bogus"])).is_err());
        let a = parse(&list(&["in", "out", "--space-hints", "h.txt", "--only-hinted"])).unwrap();
        assert_eq!((a.space_hints, a.only_hinted), (Some(PathBuf::from("h.txt")), true));
        assert!(parse(&list(&["in", "out", "--only-hinted"])).is_err(), "it needs the file");
        let a = parse(&list(&["in", "out", "--table-heap", "--only-hashes", "h.txt"])).unwrap();
        assert_eq!((a.table_heap, a.only_hashes), (true, Some(PathBuf::from("h.txt"))));
    }
}
