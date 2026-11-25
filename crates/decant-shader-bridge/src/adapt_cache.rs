//! The offline adaptation cache: for each layout vkd3d-proton actually wrote (a `.layout` file), the
//! FINAL module (translated, rewritten for that layout by [`Session::adapt`], then put at the
//! Direct3D registers by `decant_shader_locate`) is prepared ahead of time, so the bridge
//! (`decant_shader_serve`) only drops it in during play.
//!
//! Unlike the translation cache (`decant_shader_cache`), adaptation depends on the layout: an entry is
//! `<pack number>.<layout digest>.spv`. Two pipelines of the same signature share one entry, and the
//! work is done once.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use sha1::{Digest, Sha1};

use crate::adapt::Layout;
use crate::py::{self, Failure};
use crate::serve::{layout_digest, read_layout};
use crate::session::Session;

/// One distinct `(shader, layout)` pair of a capture.
pub struct Entry {
    pub name: String,
    pub pack_id: i64,
    pub layout: Layout,
    pub digest: String,
}

/// The pairs of `capture` whose `.dxbc` holds console code, one per distinct `(pack, layout)`, in
/// the order of the `.layout` file names.
pub fn entries(capture: &Path, index: &HashMap<String, i64>) -> Result<Vec<Entry>, Failure> {
    let mut layouts: Vec<PathBuf> = std::fs::read_dir(capture)
        .map_err(|e| Failure::io(&e, capture))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "layout"))
        .collect();
    layouts.sort();
    let (mut seen, mut rows) = (HashSet::new(), Vec::new());
    for layout_file in layouts {
        let name = layout_file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let Ok(blob) = std::fs::read(capture.join(format!("{name}.dxbc"))) else { continue };
        // An ordinary container is left to vkd3d-proton.
        if memchr::memmem::find(&blob, b"XDAT").is_none() {
            continue;
        }
        let Some(&pack_id) = index.get(&decant_extract_shaders::output::hex(&Sha1::digest(&blob))) else { continue };
        let text = std::fs::read(&layout_file).map_err(|e| Failure::io(&e, &layout_file))?;
        let (layout, _) = read_layout(&String::from_utf8_lossy(&text))?;
        let digest = layout_digest(&layout);
        if seen.insert((pack_id, digest.clone())) {
            rows.push(Entry { name, pack_id, layout, digest });
        }
    }
    Ok(rows)
}

/// Prepares one entry and publishes it atomically. Gives `deja`, `ok` or `echec <step> : <reason>`.
pub fn build_one(session: &Session, entry: &Entry, cache: &Path) -> String {
    let stem = format!("{}.{}", entry.pack_id, entry.digest);
    let out = cache.join(format!("{stem}.spv"));
    if out.exists() {
        return "deja".into();
    }
    let work = cache.join(".work");
    let partial = cache.join(format!("{stem}.{}.partial", std::process::id()));
    let result = (|| {
        std::fs::create_dir_all(&work).map_err(|e| ("decant_shader_adapt", e.to_string()))?;
        let (module, _) = session.adapt(entry.pack_id, &entry.layout, &work.join(&stem)).map_err(|f| ("decant_shader_adapt", f.stderr().to_owned()))?;
        let adapted = work.join(format!("{stem}_adapte.spv"));
        std::fs::write(&adapted, module).map_err(|e| ("decant_shader_adapt", e.to_string()))?;
        let mut locate = session.config.tool("decant_shader_locate", &session.config.locate);
        let run = locate.arg(entry.pack_id.to_string()).arg(&adapted).arg(&partial).output().map_err(|e| ("decant_shader_locate", e.to_string()))?;
        if !run.status.success() {
            let shown = if run.stderr.is_empty() { &run.stdout } else { &run.stderr };
            return Err(("decant_shader_locate", String::from_utf8_lossy(shown).into_owned()));
        }
        // The same optional spirv-opt step as the bridge (DECANT_SHADER_SPIRV_OPT): the entry is already
        // optimised and the bridge drops it as is. A failure keeps the module as it was.
        crate::spirvopt::optimize(&partial);
        // Renamed whole, so a reader never sees half a file.
        std::fs::rename(&partial, &out).map_err(|e| ("publication", e.to_string()))
    })();
    match result {
        Ok(()) => "ok".into(),
        Err((step, reason)) => {
            let _ = std::fs::remove_file(&partial);
            format!("echec {step} : {}", py::tail(py::strip(&reason), 200))
        }
    }
}

/// The JSON summary of a run, as `adapt_cache.json` holds it.
pub fn summary(cache: &Path, rows: usize, outcomes: &[String]) -> serde_json::Value {
    let entries = std::fs::read_dir(cache)
        .map(|d| d.filter_map(Result::ok).filter(|e| e.path().extension().is_some_and(|x| x == "spv")).count())
        .unwrap_or(0);
    let mut counts = serde_json::Map::new();
    for outcome in outcomes {
        let word = outcome.split(' ').next().unwrap_or_default();
        let n = counts.get(word).and_then(serde_json::Value::as_i64).unwrap_or(0);
        counts.insert(word.to_owned(), (n + 1).into());
    }
    let failures: Vec<&String> = outcomes.iter().filter(|o| o.starts_with("echec")).take(20).collect();
    serde_json::json!({
        "entries": entries,
        "distinct_pairs": rows,
        "outcomes": counts,
        "spirv_opt": crate::spirvopt::tools(|name| std::env::var_os(name)).is_some(),
        "failures": failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_are_reduced_to_one_per_shader_and_layout() {
        let dir = tempfile::tempdir().unwrap();
        let console = b"DXBC....XDAT console".to_vec();
        let ordinary = b"DXBC ordinary".to_vec();
        let layout = "flags 4096\nbinding 2 0 188 1 0 0 0 0 0 0\n";
        for (name, blob, text) in [
            ("a", &console, layout),
            ("b", &console, layout),
            ("c", &console, "binding 2 0 5 1 0 0 0 0 0 1\n"),
            ("d", &ordinary, layout),
            ("e", &b"XDAT not in the game".to_vec(), layout),
        ] {
            std::fs::write(dir.path().join(format!("{name}.dxbc")), blob).unwrap();
            std::fs::write(dir.path().join(format!("{name}.layout")), text).unwrap();
        }
        std::fs::write(dir.path().join("f.layout"), layout).unwrap();
        let index = HashMap::from([(decant_extract_shaders::output::hex(&Sha1::digest(&console)), 7)]);
        let rows = entries(dir.path(), &index).unwrap();
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["a", "c"]);
        assert!(rows.iter().all(|r| r.pack_id == 7));
        assert_ne!(rows[0].digest, rows[1].digest);
    }

    #[test]
    fn the_summary_counts_outcomes_by_their_first_word() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("1.abc.spv"), b"m").unwrap();
        let outcomes = ["ok".to_owned(), "echec decant_shader_locate : x".to_owned(), "deja".to_owned(), "ok".to_owned()];
        let value = summary(dir.path(), 4, &outcomes);
        assert_eq!(value["entries"], 1);
        assert_eq!(crate::describe::compact(&value["outcomes"]), r#"{"ok": 2, "echec": 1, "deja": 1}"#);
        assert_eq!(value["failures"], serde_json::json!(["echec decant_shader_locate : x"]));
    }
}
