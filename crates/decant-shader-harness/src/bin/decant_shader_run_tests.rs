//! Replays the on screen tests of DecantShader, as `src/shaders/decant_shader/harness/run_tests.py` does:
//! translates game shaders (by pack number), runs them on the GPU with `decant_shader_run` and compares
//! the image with the result computed independently.
//!
//! Usage: `decant_shader_run_tests`. Writes to `harness/out`, which only holds files made from
//! title files and must not be shared. See `decant_shader_harness::Harness` for the programs it runs.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use decant_shader_harness::{Harness, constants, pad, run_text, tail160, verdict};

struct Driver {
    harness: Harness,
    out: PathBuf,
    vertex: PathBuf,
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// `translate()`: `decant_shader_one` into `out/t<n>`, then the module copied to `out/<n>.spv`.
fn translate(harness: &Harness, out: &Path, pack_id: u32) -> Result<PathBuf, String> {
    let work = out.join(format!("t{pack_id}"));
    let mut command = harness.config.tool("decant_shader_one", &harness.config.one);
    command.arg(&work).arg(pack_id.to_string());
    let ran = run_text(&mut command).map_err(|e| e.to_string())?;
    if ran.code != 0 {
        return Err(format!("subprocess.CalledProcessError: decant_shader_one {pack_id} returned non-zero exit status {}.", ran.code));
    }
    let module = out.join(format!("{pack_id}.spv"));
    let source = work.join("0.spv");
    fs::copy(&source, &module).map_err(io(&source))?;
    Ok(module)
}

impl Driver {
    /// `run()`: one test, printed on one line. True when the checker accepts the image.
    fn run(&self, name: &str, pixel: &Path, checker: &str, extra: &[OsString], check_args: &[&str]) -> Result<bool, String> {
        let image = self.out.join(format!("{name}.f32"));
        let _ = fs::remove_file(&image);
        let mut bench = std::process::Command::new(&self.harness.run_bench);
        bench.arg(&self.vertex).arg(pixel).arg(&image).args(extra);
        let bench = run_text(&mut bench).map_err(io(&self.harness.run_bench))?;
        if bench.code != 0 || !image.exists() {
            let text = if bench.stderr.is_empty() { &bench.stdout } else { &bench.stderr };
            println!("{} ECHEC du banc (code {}) : {}", pad(name, 34), bench.code, tail160(text));
            return Ok(false);
        }
        let mut check = self.harness.checker(checker);
        check.arg(&image).args(check_args);
        let checked = run_text(&mut check).map_err(|e| e.to_string())?;
        let (text, passed) = verdict(&checked);
        println!("{} {text}", pad(name, 34));
        Ok(passed)
    }
}

fn texture_options(zero: &Path) -> Vec<OsString> {
    ["--texture", "63", "111", "--buffer", "119"].iter().map(OsString::from).chain([zero.as_os_str().to_owned()]).chain(["--push", "0", "32768"].iter().map(OsString::from)).collect()
}

fn main_inner() -> Result<bool, String> {
    let harness = Harness::from_env();
    let out = harness.out();
    match fs::create_dir(&out) {
        Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => return Err(format!("{}: {e}", out.display())),
        _ => {}
    }
    let vertex = translate(&harness, &out, 442)?; // VS_Blit
    let constants_path = out.join("constantes.bin");
    fs::write(&constants_path, constants()).map_err(io(&constants_path))?;
    let zero = out.join("zero.bin");
    fs::write(&zero, vec![0u8; 4096]).map_err(io(&zero))?;
    let driver = Driver { harness, out: out.clone(), vertex };
    let s = |v: &str| OsString::from(v);
    let mut ok = Vec::new();
    let blit = translate(&driver.harness, &out, 25345)?; // PS_BlitDiffuse
    ok.push(driver.run("copie de couleur", &blit, "check_blit", &[], &[])?);
    let colour = translate(&driver.harness, &out, 25343)?; // PS_BlitColour
    let extra = [s("--buffer"), s("60"), constants_path.into_os_string(), s("--push"), s("1"), s("64")];
    ok.push(driver.run("couleur x tampon de constantes", &colour, "check_blit", &extra, &["0.5", "2.0", "0.25", "1.0"])?);
    let texture = translate(&driver.harness, &out, 11584)?; // PS_BlitTransparent
    ok.push(driver.run("echantillonnage d'une texture", &texture, "check_texture", &texture_options(&zero), &[])?);
    // The same pixel shader, linked to the vertex shader by semantic: it must read the real texture coordinate.
    let linked = out.join("11584_lie.spv");
    let mut link = driver.harness.config.tool("decant_shader_link", &driver.harness.link);
    link.current_dir(std::env::current_dir().unwrap_or_default());
    link.arg("442").arg("11584").arg(out.join("11584.spv")).arg(&linked);
    let ran = run_text(&mut link).map_err(|e| e.to_string())?;
    if ran.code != 0 {
        return Err(format!("subprocess.CalledProcessError: decant_shader_link returned non-zero exit status {}.", ran.code));
    }
    ok.push(driver.run("texture, entrees reliees", &linked, "check_texture", &texture_options(&zero), &["--lie"])?);
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
