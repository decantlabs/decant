//! The program vkd3d-proton runs when it creates a pipeline (the DecantShader hook of
//! `third_party/vkd3d-proton/vkd3d-decant-shader.patch`, variable `VKD3D_DECANT_SHADER_COMMAND`), as
//! `src/shaders/decant_shader/harness/decant_shader_serve.py` does.
//!
//! It receives a shader hash, reads the container and the layout vkd3d-proton has just written to
//! `VKD3D_DECANT_SHADER_DIR`, finds the game shader, translates it, rewrites it for that layout and
//! drops the module into `VKD3D_SHADER_OVERRIDE`. A `<hash>.log` journal is written next to the
//! container, line for line as the Python writes it.
//!
//! Unlike the Python, the intermediate files (the layout as JSON, the adapted module before
//! `decant_shader_locate`, the translation folder) live in a scratch folder of this call only, outside
//! `VKD3D_DECANT_SHADER_DIR`, removed when the call ends. Two calls running at once, even for the same
//! hash, no longer share them.

use std::collections::HashMap;
use std::path::Path;

use sha1::{Digest, Sha1};

use crate::adapt::{Layout, Target};
use crate::py::{self, Failure};
use crate::session::Session;
use crate::shaders::Config;

/// `vkd3d_shader_descriptor_type` to register letter.
fn letter(kind: i64) -> Option<char> {
    match kind {
        1 => Some('b'),
        2 => Some('t'),
        3 => Some('u'),
        4 => Some('s'),
        _ => None,
    }
}

/// `vkd3d_shader_binding_flag`.
const BINDLESS: i64 = 8;
const IMAGE: i64 = 2;
/// `VKD3D_SHADER_INTERFACE_HEAP_LOWERING`.
const HEAP_LOWERING: i64 = 0x1000;
/// `VKD3D_SHADER_TABLES_VIRTUAL_DESCRIPTOR_SET_BASE`.
const TABLES_SET_BASE: i64 = 0;

/// SHA-1 of every game container to its pack number, computed once and kept in `harness/out`, in
/// the same JSON as the Python so either version reuses the other's file. The number comes from
/// the shader's archive position (`shaders.csv`), not from its rank among the items.
pub fn container_index(session: &Session) -> Result<HashMap<String, i64>, Failure> {
    let cache = session.config.root().join("harness").join("out").join("index_conteneurs_pack.json");
    if cache.exists() {
        let text = std::fs::read_to_string(&cache).map_err(|e| Failure::io(&e, &cache))?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| Failure::error("json.decoder.JSONDecodeError", &e.to_string()))?;
        let object = value.as_object().ok_or_else(|| Failure::error("AttributeError", "the container index is not a JSON object"))?;
        return Ok(object.iter().filter_map(|(k, v)| Some((k.clone(), v.as_i64()?))).collect());
    }
    let mut index = HashMap::new();
    let mut order = Vec::new();
    for n in 0..session.positions()?.len() as i64 {
        let digest = decant_extract_shaders::output::hex(&Sha1::digest(session.container(n)?));
        if !index.contains_key(&digest) {
            index.insert(digest.clone(), n);
            order.push((digest, n));
        }
    }
    let text = format!("{{{}}}", order.iter().map(|(k, n)| format!("\"{k}\": {n}")).collect::<Vec<_>>().join(", "));
    let parent = cache.parent().expect("the cache has a folder");
    std::fs::create_dir_all(parent).map_err(|e| Failure::io(&e, parent))?;
    // Written whole then renamed, so a call running at the same time never reads half a file.
    let partial = parent.join(format!("index_conteneurs_pack.{}.partial", std::process::id()));
    std::fs::write(&partial, text).map_err(|e| Failure::io(&e, &partial))?;
    std::fs::rename(&partial, &cache).map_err(|e| Failure::io(&e, &cache))?;
    Ok(index)
}

/// `kind, space, ... = map(int, fields[1:])` for the ten numbers of a `binding` line.
fn ten(fields: &[&str]) -> Result<[i64; 10], Failure> {
    let mut values = [0i64; 10];
    for (i, field) in fields.iter().take(11).enumerate() {
        let value = py::int(field)?;
        if i == 10 {
            return Err(Failure::value("too many values to unpack (expected 10)".into()));
        }
        values[i] = value;
    }
    if fields.len() < 10 {
        return Err(Failure::value(format!("not enough values to unpack (expected 10, got {})", fields.len())));
    }
    Ok(values)
}

/// Reads the `.layout` vkd3d-proton wrote: the layout for `adapt`, and the lines left out.
pub fn read_layout(text: &str) -> Result<(Layout, Vec<String>), Failure> {
    let lines = py::splitlines(text);
    let mut interface_flags = 0;
    if let Some(line) = lines.iter().find(|l| l.starts_with("flags ")) {
        let fields = py::split(line);
        interface_flags = py::int(fields.get(1).ok_or_else(|| Failure::error("IndexError", "list index out of range"))?)?;
    }
    let (mut layout, mut skipped) = (Layout::new(), Vec::new());
    for line in lines {
        let fields = py::split(line);
        let first = fields.first().ok_or_else(|| Failure::error("IndexError", "list index out of range"))?;
        if *first != "binding" {
            continue;
        }
        let [kind, space, register, count, table, offset, _visibility, flags, mut set, mut binding] = ten(&fields[1..])?;
        // 0xFFFFFFFF: the whole heap.
        let Some(letter) = letter(kind).filter(|_| space == 0 && register != 0xFFFF_FFFF) else {
            skipped.push(line.to_owned());
            continue;
        };
        if flags & BINDLESS != 0 {
            // Same rule as vkd3d-proton (dxil.c, dxil_remap_inner, "heap lowering"): a non array
            // resource of a descriptor table becomes an ordinary binding, set = table number,
            // binding = offset in the table. One entry per nature (buffer, image): the image is taken.
            if interface_flags & HEAP_LOWERING == 0 || count != 1 || flags & IMAGE == 0 {
                skipped.push(line.to_owned());
                continue;
            }
            (set, binding) = (TABLES_SET_BASE + table, offset);
        }
        let target = Target {
            set: u32::try_from(set).map_err(|_| Failure::error("struct.error", "argument out of range"))?,
            binding: u32::try_from(binding).map_err(|_| Failure::error("struct.error", "argument out of range"))?,
            uniform: kind == 1,
        };
        let key = format!("{letter}{register}");
        // A dict keeps the first position of a key and its last value.
        match layout.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = target,
            None => layout.push((key, target)),
        }
    }
    Ok((layout, skipped))
}

/// `repr()` of the layout dict.
pub fn layout_repr(layout: &Layout) -> String {
    let entries: Vec<String> = layout
        .iter()
        .map(|(k, t)| {
            let genre = if t.uniform { ", 'genre': 'uniforme'" } else { "" };
            format!("{}: {{'set': {}, 'binding': {}{genre}}}", py::repr_str(k), t.set, t.binding)
        })
        .collect();
    format!("{{{}}}", entries.join(", "))
}

/// Stable digest of a layout (truncated SHA-1), computed identically by the Python bridge
/// (`decant_shader_serve.py`, `layout_digest`). Key of the offline adaptation cache
/// (`decant_shader_adapt_cache.py`): two pipelines of the same signature share one entry.
pub fn layout_digest(layout: &Layout) -> String {
    let mut parts: Vec<String> = layout
        .iter()
        .map(|(k, t)| format!("{k}={}:{}:{}", t.set, t.binding, t.uniform as u8))
        .collect();
    parts.sort();
    decant_extract_shaders::output::hex(&Sha1::digest(parts.join(";").as_bytes()))[..16].to_string()
}

/// One journal line for a step: `<script> : <stdout> <last 400 characters of stderr>`.
fn step_line(script: &str, stdout: &str, stderr: &str) -> String {
    format!("{script} : {} {}", py::strip(stdout), py::tail(py::strip(stderr), 400))
}

/// Serves one hash. Gives the journal lines, and the `SystemExit` that ends the run, if any (the
/// Python writes the journal and then exits with status 1).
pub fn serve(config: Config, name: &str, work: &Path, override_dir: &Path) -> (Vec<String>, Option<Failure>) {
    let mut log = Vec::new();
    match serve_in(config, name, work, override_dir, &mut log) {
        Ok(()) => (log, None),
        Err(exit @ Failure::Exit(_)) => (log, Some(exit)),
        // The journal must say why vkd3d-proton will not receive a module.
        Err(Failure::Error { repr, .. }) => {
            log.push(format!("ECHEC : {repr}"));
            (log, None)
        }
    }
}

fn serve_in(config: Config, name: &str, work: &Path, override_dir: &Path, log: &mut Vec<String>) -> Result<(), Failure> {
    let session = Session::open(config)?;
    let path = work.join(format!("{name}.dxbc"));
    let blob = std::fs::read(&path).map_err(|e| Failure::io(&e, &path))?;
    if memchr::memmem::find(&blob, b"XDAT").is_none() {
        log.push("conteneur ordinaire (pas de code console) : laisse a vkd3d-proton".into());
        return Ok(());
    }
    let digest = decant_extract_shaders::output::hex(&Sha1::digest(&blob));
    let pack_id = *container_index(&session)?.get(&digest).ok_or_else(|| Failure::runtime("conteneur absent du jeu"))?;
    let path = work.join(format!("{name}.layout"));
    let text = std::fs::read(&path).map_err(|e| Failure::io(&e, &path))?;
    let (layout, skipped) = read_layout(&String::from_utf8_lossy(&text))?;
    let skipped_repr: Vec<String> = skipped.iter().map(|s| py::repr_str(s)).collect();
    log.push(format!("shader {pack_id} ; disposition de vkd3d-proton : {} ; non gerees : [{}]", layout_repr(&layout), skipped_repr.join(", ")));

    let final_module = override_dir.join(format!("{name}.spv"));
    // Offline adaptation cache (decant_shader_adapt_cache.py): if the final module for this layout is already
    // prepared, drop it as is — no translation, adaptation or relocation. Otherwise fall back to the work below.
    if let Some(dir) = std::env::var_os("DECANT_SHADER_ADAPT_CACHE") {
        let ready = Path::new(&dir).join(format!("{pack_id}.{}.spv", layout_digest(&layout)));
        if ready.exists() {
            std::fs::copy(&ready, &final_module).map_err(|e| Failure::io(&e, &ready))?;
            let shown = ready.file_name().unwrap_or(ready.as_os_str()).to_string_lossy();
            log.push(format!("cache d'adaptation : {shown}"));
            return Ok(());
        }
    }

    let scratch = tempfile::Builder::new().prefix("decant_shader_serve_").tempdir().map_err(|e| Failure::io(&e, &std::env::temp_dir()))?;
    let adapted = scratch.path().join(format!("{name}_adapte.spv"));

    let (stdout, stderr) = match session.adapt(pack_id, &layout, &scratch.path().join("adapt")) {
        Ok((module, report)) => match std::fs::write(&adapted, module) {
            Ok(()) => (report.iter().map(|l| format!("   {l}\n")).collect::<String>(), None),
            Err(e) => (String::new(), Some(Failure::io(&e, &adapted))),
        },
        Err(failure) => (String::new(), Some(failure)),
    };
    log.push(step_line("decant_shader_adapt.py", &stdout, stderr.as_ref().map_or("", Failure::stderr)));
    if stderr.is_some() {
        return Err(Failure::runtime("decant_shader_adapt.py a echoue"));
    }

    let mut locate = session.config.tool("decant_shader_locate", &session.config.locate);
    locate.arg(pack_id.to_string()).arg(&adapted).arg(&final_module);
    let run = locate.output().map_err(|e| Failure::io(&e, Path::new(&Config::program("decant_shader_locate", &session.config.locate))))?;
    let text = |bytes: &[u8]| py::universal_newlines(&String::from_utf8_lossy(bytes));
    log.push(step_line("decant_shader_locate.py", &text(&run.stdout), &text(&run.stderr)));
    if !run.status.success() {
        return Err(Failure::runtime("decant_shader_locate.py a echoue"));
    }
    // Optional spirv-opt step (DECANT_SHADER_SPIRV_OPT, off by default). Not on an adaptation cache entry: the
    // cache is prepared with or without it, once (decant_shader_adapt_cache.py).
    if let Some(line) = crate::spirvopt::optimize(&final_module) {
        log.push(line);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_lines_follow_the_vkd3d_rules() {
        let text = "stage 16\nflags 4096\n\
            binding 1 0 27 1 0 0 0 0 103 0\n\
            binding 2 0 188 1 0 0 0 0 0 0\n\
            binding 2 1 3 1 0 0 0 0 0 1\n\
            binding 2 0 4294967295 1 0 0 0 0 0 2\n\
            binding 2 0 5 1 2 7 0 10 9 9\n\
            binding 2 0 6 1 2 7 0 8 9 9\n\
            binding 4 0 0 1 0 0 0 0 101 0\n\
            push 0 0 0 0 16\n";
        let (layout, skipped) = read_layout(text).unwrap();
        assert_eq!(
            layout_repr(&layout),
            "{'b27': {'set': 103, 'binding': 0, 'genre': 'uniforme'}, 't188': {'set': 0, 'binding': 0}, 't5': {'set': 2, 'binding': 7}, 's0': {'set': 101, 'binding': 0}}"
        );
        assert_eq!(skipped, ["binding 2 1 3 1 0 0 0 0 0 1", "binding 2 0 4294967295 1 0 0 0 0 0 2", "binding 2 0 6 1 2 7 0 8 9 9"]);
        // Without heap lowering, a bindless resource stays out.
        let (layout, skipped) = read_layout("binding 2 0 5 1 2 7 0 10 9 9\n").unwrap();
        assert!(layout.is_empty());
        assert_eq!(skipped.len(), 1);
    }

    #[test]
    fn layout_digest_matches_the_python_bridge() {
        let text = "flags 4096\n\
            binding 1 0 27 1 0 0 0 0 103 0\n\
            binding 2 0 188 1 0 0 0 0 0 0\n\
            binding 2 0 5 1 2 7 0 10 9 9\n\
            binding 4 0 0 1 0 0 0 0 101 0\n";
        let (layout, _) = read_layout(text).unwrap();
        // Same value as decant_shader_serve.py layout_digest on the same layout (keeps the two bridges in step).
        assert_eq!(layout_digest(&layout), "600a08313d0f48f1");
        // Order of the lines does not change the digest.
        let other = "flags 4096\n\
            binding 4 0 0 1 0 0 0 0 101 0\n\
            binding 2 0 5 1 2 7 0 10 9 9\n\
            binding 1 0 27 1 0 0 0 0 103 0\n\
            binding 2 0 188 1 0 0 0 0 0 0\n";
        assert_eq!(layout_digest(&read_layout(other).unwrap().0), "600a08313d0f48f1");
    }

    #[test]
    fn malformed_layouts_fail_like_python() {
        assert_eq!(read_layout("binding 1 2\n").unwrap_err().stderr(), "ValueError: not enough values to unpack (expected 10, got 2)");
        assert_eq!(read_layout("binding 1 2 3 4 5 6 7 8 9 10 11\n").unwrap_err().stderr(), "ValueError: too many values to unpack (expected 10)");
        assert_eq!(read_layout("binding x\n").unwrap_err().stderr(), "ValueError: invalid literal for int() with base 10: 'x'");
        assert_eq!(read_layout("stage 1\n\n").unwrap_err().stderr(), "IndexError: list index out of range");
    }

    #[test]
    fn journal_lines_strip_and_cut_like_python() {
        assert_eq!(step_line("a.py", "   x\n   y\n", ""), "a.py : x\n   y ");
        let long = "e".repeat(500);
        assert_eq!(step_line("a.py", "", &format!("{long}\n")), format!("a.py :  {}", "e".repeat(400)));
    }
}
