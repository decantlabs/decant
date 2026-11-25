//! `d3d12x_vtprobe`: measures which D3D12 vtable slots the modules of a console package call, to
//! derive the console method order (see `vtprobe.rs`). Read only: files are parsed as data.
//!
//! Usage: `d3d12x_vtprobe <module.dll>... --pc-header <d3d12.h> [--out <folder>]`
//!
//! Writes `vtprobe.json` (every call site) and `vtprobe.md` (one row per interface and slot, with
//! the PC method at the same slot and the interface IDs seen as arguments).

#[allow(dead_code)]
mod pe;
mod vtprobe;

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut modules = Vec::new();
    let mut header = None;
    let mut out = PathBuf::from("vtprobe");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            // `--dump <module> <rva hex>`: the instructions before a call site, to check a slot by hand.
            "--dump" => {
                let (Some(m), Some(rva)) = (args.next(), args.next()) else { return usage() };
                let Ok(rva) = u64::from_str_radix(rva.trim_start_matches("0x"), 16) else { return usage() };
                return match pe::Pe::open(std::path::Path::new(&m)) {
                    Ok(pe) => {
                        vtprobe::dump(&pe, rva);
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("{m}: {e}");
                        ExitCode::FAILURE
                    }
                };
            }
            "--pc-header" => header = args.next().map(PathBuf::from),
            "--out" => match args.next() {
                Some(d) => out = PathBuf::from(d),
                None => return usage(),
            },
            _ => modules.push(PathBuf::from(a)),
        }
    }
    let (Some(header), false) = (header, modules.is_empty()) else { return usage() };
    let pc = match std::fs::read_to_string(&header) {
        Ok(t) => vtprobe::PcHeader::parse(&t),
        Err(e) => {
            eprintln!("{}: {e}", header.display());
            return ExitCode::FAILURE;
        }
    };
    let mut sites = Vec::new();
    for m in &modules {
        let name = m.file_name().map_or_else(|| m.display().to_string(), |n| n.to_string_lossy().into_owned());
        match pe::Pe::open(m) {
            Ok(pe) => {
                let typed = vtprobe::probe(&pe, &name);
                let untyped = vtprobe::probe_untyped(&pe, &name, &pc);
                eprintln!("{name}: {} typed vtable calls, {} calls with a known interface ID", typed.len(), untyped.len());
                sites.extend(typed);
                sites.extend(untyped);
            }
            Err(e) => eprintln!("{name}: {e}"),
        }
    }
    let rows = vtprobe::aggregate(&sites, &pc);
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("{}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    let json = serde_json::to_string_pretty(&serde_json::json!({ "sites": sites, "slots": rows })).unwrap_or_default();
    let mut md = String::from("# D3D12 vtable slots measured in the package\n\n| Interface | Slot | Calls | PC method at this slot | Interface IDs as arguments | rdx on stack | Sites |\n|---|---|---|---|---|---|---|\n");
    for r in &rows {
        let _ = writeln!(md, "| {} | {} | {} | {} | {} | {} | {} |", r.iface, r.slot, r.calls, r.pc_method.as_deref().unwrap_or("-"), r.iids.join(", "), if r.rdx_stack { "yes" } else { "" }, r.sites.join(" "));
    }
    for (file, text) in [("vtprobe.json", json), ("vtprobe.md", md)] {
        if let Err(e) = std::fs::write(out.join(file), text) {
            eprintln!("{file}: {e}");
            return ExitCode::FAILURE;
        }
    }
    eprintln!("{} slots, report in {}", rows.len(), out.display());
    ExitCode::SUCCESS
}

fn usage() -> ExitCode {
    eprintln!("usage: d3d12x_vtprobe <module.dll>... --pc-header <d3d12.h> [--out <folder>]");
    ExitCode::from(2)
}
