//! The translation cache, as `src/shaders/decant_shader/tools/decant_shader_cache.py` builds it: one module
//! per pack shader, translated ahead of time, for `decant_shader_serve`. See
//! [`decant_shader_batch::cache`].
//!
//! Usage: `decant_shader_cache <measure> <cache>`. The pack comes from `src/shaders/out`, the probe
//! whose hash is noted from `DECANT_SHADER_PROBE` or `src/shaders/decant_shader` (see `decant_shader_core::paths`).

use std::path::Path;
use std::process::ExitCode;

use decant_shader_batch::cache::{self, Info};
use decant_shader_core::{paths, py};

/// The docstring the Python prints when the arguments are wrong.
const USAGE: &str = "Cache de traduction : un module par shader du pack, traduit a l'avance, pour que le pont vers vkd3d-proton
(decant_shader_serve) n'ait plus a lancer la sonde pendant le jeu.

usage : decant_shader_cache <mesure> <cache>
  <mesure> : dossier ecrit par decant_shader_batch avec --placeholders (les options de decant_shader_one), lance avec
             DECANT_SHADER_SOURCES=1 et DECANT_SHADER_NO_MIP_STATS=1 dans l'environnement (comme le pont) :
               set DECANT_SHADER_SOURCES=1 & set DECANT_SHADER_NO_MIP_STATS=1
               decant_shader_batch <mesure> 0 1 2 5 13 --metadata ../out/decant_shader_metadata.jsonl --placeholders
  <cache>  : dossier cree ou complete : <numero du pack>.spv et <numero du pack>.src pour chaque shader traduit,
             plus cache.json (empreinte de la sonde et du pack, nombre de modules).

Le pont lit ce dossier quand DECANT_SHADER_SPV_CACHE le designe, et retombe sur decant_shader_one pour un shader absent.
Tout ce qu'il contient est derive du jeu : a garder localement, jamais dans le depot.";

fn run(batch: &Path, cache_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(cache_dir).map_err(|e| format!("{}: {e}", cache_dir.display()))?;
    let pack = paths::out().join("shaders_gpu.pack");
    let buckets = cache::stage_indices(&pack)?;
    let copied = cache::fill(batch, cache_dir, &buckets)?;
    let probe = paths::probe();
    let info = Info {
        modules: cache::module_count(cache_dir),
        copied,
        pack_sha256: cache::digest(&pack)?,
        probe_sha256: if probe.exists() { Some(cache::digest(&probe)?) } else { None },
    };
    let path = cache_dir.join("cache.json");
    std::fs::write(&path, py::native_newlines(&info.file())).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("{}", info.line());
    if copied.missing_src > 0 {
        eprintln!("des modules sans .src : la mesure n'a pas ete lancee avec DECANT_SHADER_SOURCES=1");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [batch, cache_dir] = args.as_slice() else {
        eprintln!("{USAGE}");
        return ExitCode::from(1);
    };
    match run(Path::new(batch), Path::new(cache_dir)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
