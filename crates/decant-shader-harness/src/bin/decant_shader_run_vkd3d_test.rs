//! Tests the Direct3D 12 to vkd3d-proton to Vulkan path with title shaders, as
//! `src/shaders/decant_shader/harness/run_vkd3d_test.py` does.
//!
//! For each test the title's DXBC containers are read from the shader archive (read only) and handed as
//! they are to a Direct3D 12 pipeline. vkd3d-proton, through its DecantShader hook, has each shader
//! translated and adapted to the layout it chose (`decant_shader_serve`), then takes it through its
//! usual override. Everything goes to `harness/out` (made from title files: not to be shared).
//!
//! Usage: `decant_shader_run_vkd3d_test`, with `DECANT_VKD3D_DLL` naming the patched `d3d12.dll` and
//! `DECANT_SHADERS_RPF` the archive.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use decant_shader_bridge::py::{self, Failure};
use decant_shader_bridge::shaders::{self, Archive};
use decant_shader_harness::{Harness, constants, fresh, pad, run_text, tail160, verdict};

const VERTEX: i64 = 442; // VS_Blit

/// One test: name, pixel shader, options of `vkd3d_test`, checker and its arguments. The layout
/// files the Python lists stay unused there too.
struct Case {
    name: &'static str,
    pixel: i64,
    options: &'static [&'static str],
    checker: &'static str,
    check_args: &'static [&'static str],
}

const CASES: [Case; 3] = [
    Case { name: "copie de couleur", pixel: 25345, options: &[], checker: "check_blit", check_args: &[] },
    Case {
        name: "tampon de constantes racine b27",
        pixel: 25343,
        options: &["--cbv", "27", "constantes.bin"],
        checker: "check_blit",
        check_args: &["0.5", "2.0", "0.25", "1.0"],
    },
    Case { name: "texture t188 en table + sampler statique", pixel: 11584, options: &["--srv", "188"], checker: "check_texture", check_args: &["--lie"] },
];

fn io(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

fn fail(f: Failure) -> String {
    f.stderr().to_owned()
}

/// The journal lines the Python repeats: failures, and the reason line of `decant_shader_adapt.py`.
fn journal_lines(text: &str) -> Vec<String> {
    py::splitlines(text)
        .into_iter()
        .filter(|line| line.contains("ECHEC") || (line.starts_with("decant_shader_adapt.py") && line.chars().skip(18).any(|c| c == ':')))
        .map(|line| format!("       {}", line.chars().take(150).collect::<String>()))
        .collect()
}

fn main_inner() -> Result<bool, String> {
    let vkd3d = PathBuf::from(std::env::var_os("DECANT_VKD3D_DLL").ok_or("KeyError: 'DECANT_VKD3D_DLL'")?);
    let harness = Harness::from_env();
    let out = harness.out();
    match fs::create_dir(&out) {
        Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => return Err(format!("{}: {e}", out.display())),
        _ => {}
    }
    let override_dir = out.join("vkd3d_override");
    fresh(&override_dir).map_err(io(&override_dir))?;
    let archive = Archive::open(&shaders::archive_path().map_err(fail)?).map_err(fail)?;
    // Pack number to archive item, by the position in shaders.csv (the pack only keeps the
    // extracted shaders, so its number can differ from the rank in the archive).
    let positions = shaders::pack_positions(&harness.config).map_err(fail)?;
    let wanted = std::iter::once(VERTEX).chain(CASES.iter().map(|c| c.pixel));
    let mut blobs: Vec<(i64, Vec<u8>)> = Vec::new();
    for n in wanted {
        let position = *py::index(&positions, n).map_err(fail)?;
        let item = archive.item_of(&positions, n).map_err(|_| format!("KeyError: {position}"))?;
        let blob = archive.container(item).map_err(fail)?;
        match blobs.iter_mut().find(|(k, _)| *k == n) {
            Some((_, b)) => *b = blob,
            None => blobs.push((n, blob)),
        }
    }
    for (n, blob) in &blobs {
        let path = out.join(format!("{n}.dxbc"));
        fs::write(&path, blob).map_err(io(&path))?;
    }
    let constants_path = out.join("constantes.bin");
    fs::write(&constants_path, constants()).map_err(io(&constants_path))?;
    // Automatic mode: nothing is prepared ahead. The hooked vkd3d-proton writes each shader's
    // container and layout into `bridge`, then runs the launcher, which puts the adapted module
    // in `override`.
    let bridge = out.join("decant");
    let serve = out.join("decant_shader_serve.cmd");
    fs::write(&serve, harness.serve_launcher()).map_err(io(&serve))?;
    let log = out.join("vkd3d.log");
    let mut ok = Vec::new();
    for case in &CASES {
        fresh(&override_dir).map_err(io(&override_dir))?;
        fresh(&bridge).map_err(io(&bridge))?;
        let image = out.join(format!("vkd3d_{}.f32", case.pixel));
        let _ = fs::remove_file(&image);
        let _ = fs::remove_file(&log);
        let options: Vec<OsString> =
            case.options.iter().map(|o| if o.ends_with(".bin") { out.join(o).into_os_string() } else { OsString::from(o) }).collect();
        let mut bench = Command::new(&harness.vkd3d_test);
        bench.arg(&vkd3d).arg(out.join(format!("{VERTEX}.dxbc"))).arg(out.join(format!("{}.dxbc", case.pixel))).arg(&image).args(&options);
        bench
            .env("VKD3D_SHADER_OVERRIDE", &override_dir)
            .env("VKD3D_DECANT_SHADER_DIR", &bridge)
            .env("VKD3D_DECANT_SHADER_COMMAND", format!("call \"{}\"", serve.display()))
            .env("DECANT_SHADER_NO_MIP_STATS", "1")
            .env("VKD3D_DEBUG", "warn")
            .env("VKD3D_LOG_FILE", &log)
            .current_dir(&out);
        let ran = run_text(&mut bench).map_err(io(&harness.vkd3d_test))?;
        let tail = tail160(&ran.stderr);
        let mut text = format!("ECHEC : {}", if tail.is_empty() { format!("code de sortie {}, pas d'image", ran.code) } else { tail.to_owned() });
        if image.exists() {
            let mut check = harness.checker(case.checker);
            check.arg(&image).args(case.check_args).current_dir(harness.config.root());
            let checked = run_text(&mut check).map_err(|e| e.to_string())?;
            text = verdict(&checked).0;
        }
        let replaced = fs::read(&log).map(|b| decant_shader_core::py::read_text(&b).matches("Overriding shader hash").count()).unwrap_or(0);
        println!("{} {text}   (shaders servis automatiquement : {replaced})", pad(case.name, 44));
        let mut journals: Vec<PathBuf> = fs::read_dir(&bridge)
            .map_err(io(&bridge))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("log")))
            .collect();
        journals.sort_by_key(|p| p.to_string_lossy().to_lowercase());
        for journal in journals {
            let bytes = fs::read(&journal).map_err(io(&journal))?;
            for line in journal_lines(&String::from_utf8_lossy(&bytes)) {
                println!("{line}");
            }
        }
        ok.push(text.contains("CONFORME") && replaced == 2);
    }
    let passed = ok.iter().filter(|&&o| o).count();
    println!("{passed} essai(s) conforme(s) sur {}", ok.len());
    Ok(passed == ok.len())
}

fn main() -> ExitCode {
    match main_inner() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_lines_are_the_ones_the_python_repeats() {
        let text = "decant_shader_adapt.py : ressource deplacee\ndecant_shader_adapt.py\nautre ligne\nECHEC de la traduction\r\ndecant_shader_adapt.py x";
        assert_eq!(journal_lines(text), ["       decant_shader_adapt.py : ressource deplacee", "       ECHEC de la traduction"]);
        let long = format!("ECHEC {}", "y".repeat(200));
        assert_eq!(journal_lines(&long)[0].chars().count(), 7 + 150);
    }
}
