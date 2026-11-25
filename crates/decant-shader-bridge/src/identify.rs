//! Ties every resource of a translated shader to its Direct3D register (b, t, s), as
//! `src/shaders/decant_shader/tools/decant_shader_identify.py` does.
//!
//! Two sources are crossed:
//! * the `<n>.src` file the probe writes with `DECANT_SHADER_SOURCES=1`: for each resource, the
//!   expression of its descriptor (table held by register sN, byte offset) and its place in the
//!   emitted module,
//! * the binding table of the shader's FSSM container: Direct3D register to table and offset.

use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;

use crate::py::{self, Failure};
use crate::shaders::Config;

/// One identified resource, the dictionary `identify()` returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resource {
    /// `genre`: the resource kind as the probe names it (`buffer`, `image`, `sampler`).
    pub kind: String,
    /// `indice`: its number among resources of that kind.
    pub index: i64,
    /// `registre`: `t188`, `b27`, or unknown.
    pub register: Option<String>,
    /// `origine`: `table`, `tas`, `tas-samplers`, `immediat` or `autre`.
    pub origin: &'static str,
    /// `tas`: the index in a directly indexed heap.
    pub heap: Option<u64>,
    /// `liaison`: the binding kind of its place in the module.
    pub binding: Option<i64>,
    /// `rang`: its element in that binding's array.
    pub rank: Option<i64>,
    /// `table` and `decalage`: the scalar register holding the table and the byte offset.
    pub table: Option<u64>,
    pub offset: Option<u64>,
}

impl Resource {
    /// `r["registre"] or r["tas"] is not None`.
    pub fn known(&self) -> bool {
        self.register.is_some() || self.heap.is_some()
    }
}

/// The FSSM binding table: register by (scalar register of the table, byte offset), the scalar
/// registers holding the resource and sampler heaps, and the sampler registers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    pub located: HashMap<(u64, u64), String>,
    pub heap: Option<u64>,
    pub sampler_heap: Option<u64>,
    pub samplers: Vec<String>,
}

fn kind_letter(kind: u32) -> Option<&'static str> {
    // 0x303 and 0x2 are seen in the corpus (u9, b502).
    match kind {
        0x300 => Some("t"),
        0x302 | 0x2 => Some("b"),
        0x303 => Some("u"),
        0x401 => Some("s"),
        _ => None,
    }
}

/// `struct.unpack_from("<{count}I" or "<Q", fs, at)`, with the error `struct` raises.
fn words<const N: usize>(fs: &[u8], at: u64) -> Result<[u32; N], Failure> {
    let need = 4 * N as u64;
    let end = at.checked_add(need).filter(|&e| e <= fs.len() as u64).ok_or_else(|| {
        Failure::error(
            "error",
            &format!("unpack_from requires a buffer of at least {} bytes for unpacking {need} bytes at offset {at} (actual buffer size is {})", at.saturating_add(need), fs.len()),
        )
    })?;
    let bytes = &fs[at as usize..end as usize];
    Ok(std::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())))
}

fn u64_at(fs: &[u8], at: u64) -> Result<u64, Failure> {
    let need = 8u64;
    let end = at + need;
    if end > fs.len() as u64 {
        return Err(Failure::error(
            "error",
            &format!("unpack_from requires a buffer of at least {end} bytes for unpacking {need} bytes at offset {at} (actual buffer size is {})", fs.len()),
        ));
    }
    Ok(u64::from_le_bytes(fs[at as usize..end as usize].try_into().unwrap()))
}

/// Reads the binding tables of an FSSM block.
pub fn table(fs: &[u8]) -> Result<Table, Failure> {
    let start = u64_at(fs, 0xF0)?;
    let [count] = words::<1>(fs, 0xFC)?;
    let mut table = Table::default();
    // First table (field 0xE8, count at 0xF8): records of five words. Seen in the corpus: kind 5
    // gives the scalar register holding the resource heap, kind 6 the sampler heap (last word).
    let first = u64_at(fs, 0xE8)?;
    let first_count = if first < fs.len() as u64 { words::<1>(fs, 0xF8)?[0] } else { 0 };
    for i in 0..u64::from(first_count) {
        let record = words::<5>(fs, first + 16 + 20 * i)?;
        if record[0] == 5 {
            table.heap = Some(u64::from(record[4] & 0xFF));
        }
        if record[0] == 6 {
            table.sampler_heap = Some(u64::from(record[4] & 0xFF));
        }
    }
    if start > fs.len() as u64 {
        return Ok(table);
    }
    for i in 0..u64::from(count) {
        let [kind, offset, register, _, flags] = words::<5>(fs, start + 16 + 20 * i)?;
        match kind_letter(kind) {
            Some("s") => table.samplers.push(format!("s{register}")),
            Some(letter) => {
                table.located.insert((u64::from((flags >> 8) & 0xFF), u64::from(offset) * 4), format!("{letter}{register}"));
            }
            None => {}
        }
    }
    Ok(table)
}

/// Matches `^ReadConst\(GetSrtResource,#[0-9a-f]+,=LoadAddressU32@(\d+)\(GetAddressResource\(GetUserData\(s(\d+)\),#0,\),#0,`
/// and gives the offset and the scalar register.
fn static_origin(origin: &str) -> Option<(u64, u64)> {
    fn digits(text: &str) -> Option<(&str, &str)> {
        let end = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
        (end > 0).then(|| text.split_at(end))
    }
    let rest = origin.strip_prefix("ReadConst(GetSrtResource,#")?;
    let end = rest.find(|c: char| !matches!(c, '0'..='9' | 'a'..='f')).unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    let rest = rest[end..].strip_prefix(",=LoadAddressU32@")?;
    let (offset, rest) = digits(rest)?;
    let rest = rest.strip_prefix("(GetAddressResource(GetUserData(s")?;
    let (register, rest) = digits(rest)?;
    rest.strip_prefix("),#0,),#0,")?;
    Some((offset.parse().ok()?, register.parse().ok()?))
}

/// `re.fullmatch(r"sampler \d+ #[0-9a-f]+", line)`.
fn immediate_sampler(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("sampler ") else { return false };
    let Some((number, value)) = rest.split_once(' ') else { return false };
    let Some(hex) = value.strip_prefix('#') else { return false };
    !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()) && !hex.is_empty() && hex.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

fn field<'a>(fields: &[&'a str], i: usize) -> Result<&'a str, Failure> {
    fields.get(i).copied().ok_or_else(|| Failure::error("IndexError", "list index out of range"))
}

/// What one data word of the push block (words before `A` in the `push A B C` line) carries on the console.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushSource {
    /// The value of scalar register `sN` (a user SGPR, or a system value the console puts in a scalar register).
    Sgpr(u32),
    /// The dispatch thread extent of an axis (0 to 2), written by the host.
    Dispatch(u8),
    /// One half of the ray query context address (0 low, 1 high).
    Raytrace(u8),
}

/// The `pushreg <word> <source>` lines of the probe: which console register each prefix data word
/// carries. Sources are `s<N>`, `dispatch.<axis>` and `raytrace.<half>`. Lines that do not parse are
/// skipped, so an old `.src` file gives an empty list.
pub fn push_registers(lines: &[String]) -> Vec<(u32, PushSource)> {
    let mut out = Vec::new();
    for line in lines {
        let Some(rest) = line.strip_prefix("pushreg ") else { continue };
        let Some((word, source)) = rest.trim().split_once(' ') else { continue };
        let Ok(word) = word.parse::<u32>() else { continue };
        let source = source.trim();
        let parsed = if let Some(n) = source.strip_prefix('s') {
            n.parse().ok().map(PushSource::Sgpr)
        } else if let Some(axis) = source.strip_prefix("dispatch.") {
            axis.parse().ok().filter(|&a| a < 3).map(PushSource::Dispatch)
        } else if let Some(half) = source.strip_prefix("raytrace.") {
            half.parse().ok().filter(|&h| h < 2).map(PushSource::Raytrace)
        } else {
            None
        };
        if let Some(source) = parsed {
            out.push((word, source));
        }
    }
    out
}

/// Ties the resources of the `.src` lines to the registers of the binding table.
pub fn identify(table: &Table, pack_id: i64, lines: &[String]) -> Result<Vec<Resource>, Failure> {
    // Origin lines describe the program before compilation, "place" lines the compiled program.
    // They are tied by rank. Recent probes also write the descriptor's source id on both sides: if
    // they differ at one rank, compilation renumbered the resources and the rank is worthless, so
    // the shader is refused instead of moving the wrong resource.
    let mut places: HashMap<(String, i64), (i64, i64)> = HashMap::new();
    let mut place_sources: HashMap<(String, i64), i64> = HashMap::new();
    let mut sources_of: Vec<((String, i64), i64)> = Vec::new();
    for line in lines {
        let fields = py::split(line);
        if line.starts_with("place ") {
            let key = (field(&fields, 1)?.to_owned(), py::int(field(&fields, 2)?)?);
            places.insert(key.clone(), (py::int(field(&fields, 3)?)?, py::int(field(&fields, 4)?)?));
            if fields.len() > 5 {
                place_sources.insert(key, py::int(fields[5])?);
            }
        } else if line.starts_with("source ") {
            let key = (field(&fields, 1)?.to_owned(), py::int(field(&fields, 2)?)?);
            let value = py::int(field(&fields, 3)?)?;
            match sources_of.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = value,
                None => sources_of.push((key, value)),
            }
        }
    }
    let mut moved: Vec<&(String, i64)> = sources_of.iter().filter(|(k, s)| place_sources.get(k).is_some_and(|p| p != s)).map(|(k, _)| k).collect();
    moved.sort();
    if !moved.is_empty() {
        let list: Vec<String> = moved.iter().map(|(k, i)| format!("({}, {i})", py::repr_str(k))).collect();
        return Err(Failure::Exit(format!("shader {pack_id} : la compilation a renumerote des ressources [{}] ; identification refusee", list.join(", "))));
    }
    let immediate = lines.iter().filter(|l| l.starts_with("sampler ") && immediate_sampler(l)).count();
    let mut out = Vec::new();
    for line in lines {
        // `ssharp` lines give the four words of an immediate sampler, not a resource.
        if line.starts_with("place ") || line.starts_with("push ") || line.starts_with("source ") || line.starts_with("ssharp ") {
            continue;
        }
        let kind = line.split(' ').next().unwrap_or("").to_owned();
        let index = py::int(field(&py::split(line), 1)?)?;
        let origin = if line.matches(' ').count() >= 2 { line.splitn(3, ' ').nth(2).unwrap_or("") } else { "" };
        let (mut register, mut how, mut heap) = (None, "autre", None);
        let matched = static_origin(origin);
        if let Some((offset, table_register)) = matched {
            how = "table";
            register = table.located.get(&(table_register, offset)).cloned();
            // No register: direct indexing of a heap (Shader Model 6.6) at a constant index. The
            // first table gives the register holding each heap, with 32 bytes per resource and 16 per sampler.
            if register.is_none() && Some(table_register) == table.heap && offset % 32 == 0 {
                (how, heap) = ("tas", Some(offset / 32));
            }
            if register.is_none() && Some(table_register) == table.sampler_heap && offset % 16 == 0 && kind == "sampler" {
                (how, heap) = ("tas-samplers", Some(offset / 16));
            }
        } else if kind == "sampler" && origin.starts_with('#') {
            how = "immediat";
            if immediate == 1 && table.samplers.len() == 1 {
                register = Some(table.samplers[0].clone());
            }
        }
        let (binding, rank) = places.get(&(kind.clone(), index)).map_or((None, None), |&(b, r)| (Some(b), Some(r)));
        out.push(Resource {
            kind,
            index,
            register,
            origin: how,
            heap,
            binding,
            rank,
            table: matched.map(|(_, t)| t),
            offset: matched.map(|(o, _)| o),
        });
    }
    Ok(out)
}

/// The module and origin record of `pack_id` from a translation cache, copied into `work` as the probe
/// would write them (`0.spv`, `0.src`); `None` when the cache does not hold this shader.
pub fn cached_sources(cache: &Path, pack_id: i64, work: &Path) -> Result<Option<Vec<String>>, Failure> {
    let (spv, src) = (cache.join(format!("{pack_id}.spv")), cache.join(format!("{pack_id}.src")));
    if !(spv.is_file() && src.is_file()) {
        return Ok(None);
    }
    std::fs::create_dir_all(work).map_err(|e| Failure::io(&e, work))?;
    std::fs::copy(&spv, work.join("0.spv")).map_err(|e| Failure::io(&e, &spv))?;
    std::fs::copy(&src, work.join("0.src")).map_err(|e| Failure::io(&e, &src))?;
    let text = std::fs::read(&src).map_err(|e| Failure::io(&e, &src))?;
    Ok(Some(py::splitlines(&String::from_utf8_lossy(&text)).into_iter().map(str::to_owned).collect()))
}

/// Translates the shader with the origin record (`decant_shader_one.py` with `DECANT_SHADER_SOURCES=1`)
/// into `work`, and gives the lines of `0.src`, or `None` when the translation failed.
pub fn sources(config: &Config, pack_id: i64, work: &Path) -> Result<Option<Vec<String>>, Failure> {
    let _ = std::fs::remove_dir_all(work);
    // The translation cache (`decant_shader_cache.py`): the module and its origin record, translated ahead of time.
    if let Some(cache) = std::env::var_os("DECANT_SHADER_SPV_CACHE")
        && let Some(lines) = cached_sources(Path::new(&cache), pack_id, work)?
    {
        return Ok(Some(lines));
    }
    let mut command = config.tool("decant_shader_one", &config.one);
    command.arg(work).arg(pack_id.to_string()).env("DECANT_SHADER_SOURCES", "1").env("DECANT_SHADER_NO_MIP_STATS", "1");
    command.stdin(Stdio::inherit()).stdout(Stdio::piped()).stderr(Stdio::piped());
    command.output().map_err(|e| Failure::io(&e, Path::new(&Config::program("decant_shader_one", &config.one))))?;
    let path = work.join("0.src");
    if !(path.exists() && work.join("0.spv").exists()) {
        return Ok(None);
    }
    let text = std::fs::read(&path).map_err(|e| Failure::io(&e, &path))?;
    Ok(Some(py::splitlines(&String::from_utf8_lossy(&text)).into_iter().map(str::to_owned).collect()))
}

/// The detail line of one resource, as the Python prints it.
pub fn detail_line(r: &Resource) -> String {
    format!(
        "   {:7} {:>3}  {:8}  registre {:6} tas {:5} liaison {}  rang {}",
        r.kind,
        r.index,
        r.origin,
        py::opt(&r.register),
        py::opt(&r.heap),
        py::opt(&r.binding),
        py::opt(&r.rank)
    )
}

/// Counts of the measurement mode, kept in the order Python's `Counter` first sees each key.
#[derive(Default)]
pub struct Measure {
    pub shaders: Vec<(&'static str, i64)>,
    pub stat: HashMap<(String, &'static str, &'static str), i64>,
    pub detail: Vec<((String, u64, u64, String), i64)>,
}

impl Measure {
    fn bump(&mut self, key: &'static str, by: i64) {
        match self.shaders.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 += by,
            None => self.shaders.push((key, by)),
        }
    }

    pub fn untranslated(&mut self) {
        self.bump("traduction non rejouee", 1);
    }

    /// Adds one identified shader. `heap_flags` is the `repr()` of its metadata field.
    pub fn add(&mut self, resources: &[Resource], heap_flags: &str) {
        self.bump("shaders", 1);
        for r in resources {
            let known = r.known();
            if !known && r.origin == "table" {
                let key = (r.kind.clone(), r.table.unwrap_or(0), r.offset.unwrap_or(0) % 32, heap_flags.to_owned());
                match self.detail.iter_mut().find(|(k, _)| *k == key) {
                    Some(entry) => entry.1 += 1,
                    None => self.detail.push((key, 1)),
                }
            }
            *self.stat.entry((r.kind.clone(), r.origin, if known { "identifie" } else { "non identifie" })).or_default() += 1;
        }
        let fixed: Vec<&Resource> = resources.iter().filter(|r| r.origin != "autre").collect();
        let all_fixed = fixed.len() == resources.len();
        self.bump("shaders dont toutes les ressources ont une origine fixe", i64::from(all_fixed));
        self.bump("... et toutes identifiees", i64::from(all_fixed && resources.iter().all(Resource::known)));
        self.bump("... sans compter les samplers immediats", i64::from(fixed.iter().all(|r| r.known() || r.origin == "immediat") && all_fixed));
    }

    /// The lines the Python prints at the end of the measurement.
    pub fn report(&self) -> Vec<String> {
        let mut lines: Vec<String> = self.shaders.iter().map(|(k, v)| format!("{v:6}  {k}")).collect();
        let mut stat: Vec<_> = self.stat.iter().collect();
        stat.sort();
        lines.extend(stat.into_iter().map(|((kind, origin, known), v)| format!("{v:6}  {kind:8} {origin:9} {known}")));
        // most_common(12): by count, ties in first seen order.
        let mut detail: Vec<_> = self.detail.iter().collect();
        detail.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        let shown: Vec<String> = detail.iter().take(12).map(|((kind, t, o, f), n)| format!("(({}, {t}, {o}, {f}), {n})", py::repr_str(kind))).collect();
        lines.push(format!("non identifies en table (genre, registre de table, decalage mod 32, drapeaux de tas) : [{}]", shown.join(", ")));
        lines
    }
}

/// The pack numbers of the valid shaders in a `results.jsonl`: rows whose `V` starts with `OK`.
pub fn valid_shaders(rows: &[serde_json::Value]) -> Result<Vec<i64>, Failure> {
    let mut valid = Vec::new();
    for row in rows {
        let ok = match row.get("V") {
            None => false,
            Some(serde_json::Value::Array(v)) => v.first().ok_or_else(|| Failure::error("IndexError", "list index out of range"))? == "OK",
            Some(_) => false,
        };
        if ok {
            let id = row.get("pack_id").ok_or_else(|| Failure::key(py::repr_str("pack_id")))?;
            valid.push(id.as_i64().ok_or_else(|| Failure::error("TypeError", "pack_id is not an integer"))?);
        }
    }
    Ok(valid)
}

/// Shaders the measurement draws: `random.seed(51)` then `random.sample(valid, min(count, len))`.
pub fn draw(valid: &[i64], count: i64) -> Result<Vec<i64>, Failure> {
    let k = count.min(valid.len() as i64);
    if k < 0 {
        return Err(Failure::value("Sample larger than population or is negative".into()));
    }
    Ok(crate::random::Random::seed(51).sample(valid, k as usize))
}


#[cfg(test)]
mod tests {
    use super::*;

    /// An invented FSSM block: the first table at 0x100 names s2 as the resource heap and s3 as
    /// the sampler heap. The binding table at 0x180 holds t188 at s4 + 8, b27 at s4 + 0, u9 at
    /// s5 + 4, the sampler s0 and one record of an unknown kind.
    fn fssm() -> Vec<u8> {
        let mut fs = vec![0u8; 0x200];
        fs[..4].copy_from_slice(b"FSSM");
        fs[0xE8..0xF0].copy_from_slice(&0x100u64.to_le_bytes());
        fs[0xF0..0xF8].copy_from_slice(&0x180u64.to_le_bytes());
        fs[0xF8..0xFC].copy_from_slice(&2u32.to_le_bytes());
        fs[0xFC..0x100].copy_from_slice(&5u32.to_le_bytes());
        let mut put = |at: usize, record: [u32; 5]| {
            for (i, w) in record.iter().enumerate() {
                fs[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
        };
        put(0x110, [5, 0, 0, 0, 0x7702]);
        put(0x124, [6, 0, 0, 0, 0x03]);
        put(0x190, [0x300, 2, 188, 0, 4 << 8]);
        put(0x1A4, [0x302, 0, 27, 0, 4 << 8]);
        put(0x1B8, [0x303, 1, 9, 0, 5 << 8]);
        put(0x1CC, [0x401, 0, 0, 0, 0]);
        put(0x1E0, [0x999, 0, 1, 0, 4 << 8]);
        fs
    }

    fn origin(offset: u32, register: u32) -> String {
        format!("ReadConst(GetSrtResource,#1f,=LoadAddressU32@{offset}(GetAddressResource(GetUserData(s{register}),#0,),#0,),#0)")
    }

    fn lines(extra: &[&str]) -> Vec<String> {
        let mut lines = vec![
            format!("image 0 {}", origin(8, 4)),
            format!("buffer 0 {}", origin(0, 4)),
            format!("buffer 1 {}", origin(4, 5)),
            format!("image 1 {}", origin(64, 2)),
            format!("sampler 0 {}", origin(32, 3)),
            "sampler 1 #ff00".to_owned(),
            format!("image 2 {}", origin(12, 4)),
            "buffer 2 GetUserData(s9)".to_owned(),
            "place image 0 1 0 7".to_owned(),
            "place buffer 0 0 3".to_owned(),
            "source image 0 7".to_owned(),
            "push 0 2 4".to_owned(),
        ];
        lines.extend(extra.iter().map(|l| l.to_string()));
        lines
    }

    #[test]
    fn tables_give_registers_and_heaps() {
        let table = table(&fssm()).unwrap();
        assert_eq!(table.heap, Some(2));
        assert_eq!(table.sampler_heap, Some(3));
        assert_eq!(table.samplers, ["s0"]);
        assert_eq!(table.located.len(), 3);
        assert_eq!(table.located[&(4, 8)], "t188");
        assert_eq!(table.located[&(5, 4)], "u9");
        assert_eq!(super::table(&fssm()[..0xF0]).unwrap_err().stderr(), "error: unpack_from requires a buffer of at least 248 bytes for unpacking 8 bytes at offset 240 (actual buffer size is 240)");
        // A binding table past the end gives only the heaps.
        let mut far = fssm();
        far[0xF0..0xF8].copy_from_slice(&0x1000u64.to_le_bytes());
        assert!(super::table(&far).unwrap().located.is_empty());
    }

    #[test]
    fn every_origin_kind_is_recognised() {
        let resources = identify(&table(&fssm()).unwrap(), 3, &lines(&[])).unwrap();
        let shown: Vec<String> = resources.iter().map(detail_line).collect();
        assert_eq!(
            shown,
            [
                "   image     0  table     registre t188   tas None  liaison 1  rang 0",
                "   buffer    0  table     registre b27    tas None  liaison 0  rang 3",
                "   buffer    1  table     registre u9     tas None  liaison None  rang None",
                "   image     1  tas       registre None   tas 2     liaison None  rang None",
                "   sampler   0  tas-samplers  registre None   tas 2     liaison None  rang None",
                "   sampler   1  immediat  registre s0     tas None  liaison None  rang None",
                "   image     2  table     registre None   tas None  liaison None  rang None",
                "   buffer    2  autre     registre None   tas None  liaison None  rang None",
            ]
        );
        assert_eq!((resources[6].table, resources[6].offset), (Some(4), Some(12)));
        // Two immediate samplers: neither is tied to the only table sampler.
        let resources = identify(&table(&fssm()).unwrap(), 3, &lines(&["sampler 2 #00aa"])).unwrap();
        assert_eq!(resources[5].register, None);
    }

    #[test]
    fn renumbered_resources_are_refused() {
        let refused = identify(&Table::default(), 12, &lines(&["source buffer 0 4", "place buffer 0 0 3 5", "source image 0 8"]));
        let expected = "shader 12 : la compilation a renumerote des ressources [('buffer', 0), ('image', 0)] ; identification refusee";
        assert_eq!(refused.unwrap_err(), Failure::Exit(expected.into()));
        let placed = vec!["place image 0 1 0 7".to_owned(), "source image 0 8".to_owned()];
        assert!(identify(&Table::default(), 1, &placed).is_err());
    }

    #[test]
    fn measure_counts_and_prints_like_python() {
        let table = table(&fssm()).unwrap();
        let mut measure = Measure::default();
        measure.untranslated();
        measure.add(&identify(&table, 3, &lines(&[])).unwrap(), "1");
        measure.add(&identify(&table, 4, &lines(&[])[..2]).unwrap(), "0");
        assert_eq!(
            measure.report(),
            [
                "     1  traduction non rejouee",
                "     2  shaders",
                "     1  shaders dont toutes les ressources ont une origine fixe",
                "     1  ... et toutes identifiees",
                "     1  ... sans compter les samplers immediats",
                "     1  buffer   autre     non identifie",
                "     3  buffer   table     identifie",
                "     2  image    table     identifie",
                "     1  image    table     non identifie",
                "     1  image    tas       identifie",
                "     1  sampler  immediat  identifie",
                "     1  sampler  tas-samplers identifie",
                "non identifies en table (genre, registre de table, decalage mod 32, drapeaux de tas) : [(('image', 4, 12, 1), 1)]",
            ]
        );
    }

    #[test]
    fn the_draw_follows_python_random() {
        let rows = shaders_rows();
        let valid = valid_shaders(&rows).unwrap();
        assert_eq!(valid, [0, 2, 3]);
        let all: Vec<i64> = (0..1000).collect();
        assert_eq!(draw(&all, 10).unwrap()[..3], [249, 858, 514]);
        assert_eq!(draw(&valid, 5).unwrap().len(), 3);
    }

    fn shaders_rows() -> Vec<serde_json::Value> {
        crate::shaders::json_lines(
            "{\"pack_id\": 0, \"V\": [\"OK\", 1]}\n{\"pack_id\": 1, \"V\": [\"ECHEC\"]}\n{\"pack_id\": 2, \"V\": [\"OK\"]}\n{\"pack_id\": 3, \"V\": [\"OK\"]}\n{\"pack_id\": 4}\n",
        )
        .unwrap()
    }
}

#[cfg(test)]
mod cache_tests {
    use super::cached_sources;

    #[test]
    fn a_cached_module_is_copied_as_the_probe_writes_it() {
        let cache = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        std::fs::write(cache.path().join("7.spv"), [3u8, 2, 0x23, 7]).unwrap();
        std::fs::write(cache.path().join("7.src"), b"a 1\r\nb 2\n").unwrap();
        let lines = cached_sources(cache.path(), 7, &work.path().join("w")).unwrap().unwrap();
        assert_eq!(lines, ["a 1", "b 2"]);
        assert_eq!(std::fs::read(work.path().join("w/0.spv")).unwrap(), [3u8, 2, 0x23, 7]);
        // A shader the cache does not hold falls back to the translation.
        assert!(cached_sources(cache.path(), 8, &work.path().join("x")).unwrap().is_none());
    }
}

#[cfg(test)]
mod push_register_tests {
    use super::{push_registers, PushSource};

    #[test]
    fn pushreg_lines_name_the_prefix_words() {
        let lines: Vec<String> = ["push 6 7 9", "pushreg 0 s8", "pushreg 1 s9", "pushreg 2 dispatch.1", "pushreg 4 raytrace.0", "pushreg 5 bogus", "pushreg x s1", "pushreg 3 dispatch.3"]
            .iter()
            .map(|l| (*l).to_owned())
            .collect();
        assert_eq!(push_registers(&lines), [(0, PushSource::Sgpr(8)), (1, PushSource::Sgpr(9)), (2, PushSource::Dispatch(1)), (4, PushSource::Raytrace(0))]);
        assert!(push_registers(&["push 0 0 0".to_owned()]).is_empty());
    }
}
