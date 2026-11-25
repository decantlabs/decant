//! Lists the translated modules of a `decant_shader_batch` measure folder with the shader each one
//! came from: pack number, name, size of the console container and its FNV-1a 64 hash (the one
//! `d3d12_xs::dxil_override::content_hash` computes over the bytecode) and SHA-1.
//!
//! Usage: `decant_shader_spv_manifest <measure folder> <stage> <output.tsv>` with the stage number of the
//! measure (0 pixel, 1 vertex, 5 compute). The measure folder holds `<stage>-<offset>/<local>.spv`,
//! and the pack number of a module is entry `offset + local` of the list of pack numbers of that
//! stage. `DECANT_SHADERS_DIR` and `DECANT_SHADERS_RPF` name the work folder and the archive, as for
//! the other tools.

use std::path::PathBuf;
use std::process::ExitCode;

use sha1::{Digest, Sha1};
use decant_shader_bridge::session::Session;
use decant_shader_bridge::shaders::Config;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [folder, stage, output] = args.as_slice() else {
        eprintln!("usage: decant_shader_spv_manifest <measure folder> <stage> <output.tsv>");
        return ExitCode::from(2);
    };
    let Ok(stage) = stage.parse::<u8>() else {
        eprintln!("the stage is a number");
        return ExitCode::from(2);
    };
    let config = Config::from_env();
    let pack = config.out().join("shaders_gpu.pack");
    let mut indices: Vec<usize> = Vec::new();
    if let Err(e) = decant_extract_shaders::pack::read(&pack, |index, _, s| {
        if s == stage {
            indices.push(index);
        }
        false
    }) {
        eprintln!("{}: {e}", pack.display());
        return ExitCode::from(1);
    }
    let session = match Session::open(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let mut rows = Vec::new();
    let Ok(rd) = std::fs::read_dir(folder) else {
        eprintln!("cannot read {folder}");
        return ExitCode::from(1);
    };
    let prefix = format!("{stage}-");
    let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&prefix))).collect();
    dirs.sort();
    for dir in dirs {
        let offset: usize = dir.file_name().and_then(|n| n.to_string_lossy().split('-').nth(1).and_then(|o| o.parse().ok())).unwrap_or(0);
        let Ok(files) = std::fs::read_dir(&dir) else { continue };
        let mut spv: Vec<(usize, PathBuf)> = files
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "spv"))
            .filter_map(|p| p.file_stem().and_then(|s| s.to_string_lossy().parse::<usize>().ok()).map(|n| (n, p.clone())))
            .collect();
        spv.sort();
        for (local, path) in spv {
            let Some(&pack_id) = indices.get(offset + local) else { continue };
            let (name, container) = match (session.name(pack_id as i64), session.container(pack_id as i64)) {
                (Ok(n), Ok(c)) => (n, c),
                _ => continue,
            };
            let sha: String = Sha1::digest(&container).iter().map(|b| format!("{b:02x}")).collect();
            rows.push(format!("{}\t{pack_id}\t{name}\t{}\t{:016x}\t{sha}", path.display(), container.len(), fnv1a64(&container)));
        }
    }
    let mut text = String::from("spv\tpack_id\tname\tcontainer_bytes\tfnv1a64\tsha1\n");
    text.push_str(&rows.join("\n"));
    text.push('\n');
    if let Err(e) = std::fs::write(output, text) {
        eprintln!("{output}: {e}");
        return ExitCode::from(1);
    }
    println!("{} modules listed", rows.len());
    ExitCode::SUCCESS
}
