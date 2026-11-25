//! Finds the console shader containers in a file or a folder of files (by the container's own
//! header) and isolates their GPU code.
//!
//! Each shader is a DXBC container whose DXIL bytecode was removed. The code already compiled for
//! the console GPU (RDNA2) is in the XDAT, XBCH, FSSM blocks.
//!
//! Usage: `extract_shaders <file or folder> [--out folder] [--files]`
//!
//! Outputs, in `--out` (default `out`):
//! * `shaders_gpu.pack`: every GPU code in one file, for the decoder test bench
//! * `shaders.csv`: one line per shader (name, type, sizes). The `position` column is the shader's
//!   position in the input (see `Shader::position`), which ties a pack index to its container.
//! * with `--files`: `dxbc/`, `fssm/` and `code/` files per shader
//!
//! The input is opened read only.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use decant_extract_shaders::container;
use decant_extract_shaders::fssm::{self, Code};
use decant_extract_shaders::{Shader, find_shaders, map_shaders, output};

struct Args {
    input: PathBuf,
    out: PathBuf,
    files: bool,
}

fn args() -> Result<Args, String> {
    let mut input = None;
    let mut out = PathBuf::from("out");
    let mut files = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => out = it.next().map(PathBuf::from).ok_or("--out needs a folder")?,
            "--fichiers" | "--files" => files = true,
            _ if input.is_none() => input = Some(PathBuf::from(arg)),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    Ok(Args { input: input.ok_or("usage: extract_shaders <file or folder> [--out folder] [--files]")?, out, files })
}

/// What one shader yields: its container, its stage, and the FSSM block with its split code.
struct Decoded {
    blob: Vec<u8>,
    stage: String,
    fssm: Option<Vec<u8>>,
    code: Option<Code>,
}

fn decode(shader: &Shader) -> Decoded {
    let blob = shader.container().to_vec();
    let (stage, _, _) = container::kind(&blob);
    let fssm = container::gpu_code(&blob).filter(|f| !f.is_empty());
    let code = fssm.as_deref().and_then(|f| fssm::split(f).ok());
    Decoded { blob, stage, fssm, code }
}

fn run(args: &Args) -> Result<u8, String> {
    eprintln!("Analyse de {}...", args.input.display());
    let shaders = find_shaders(&args.input)?;
    let shaders: Vec<&Shader> = shaders.iter().collect();
    let decoded: Vec<Decoded> = map_shaders(&shaders, decode)?;

    let io = |e: std::io::Error| e.to_string();
    fs::create_dir_all(&args.out).map_err(io)?;
    if args.files {
        for sub in ["dxbc", "fssm", "code"] {
            fs::create_dir_all(args.out.join(sub)).map_err(io)?;
        }
    }
    let mut pack = BufWriter::new(File::create(args.out.join("shaders_gpu.pack")).map_err(io)?);
    let mut table = BufWriter::new(File::create(args.out.join("shaders.csv")).map_err(io)?);
    output::csv_row(&mut table, &output::CSV_HEADER.map(str::to_owned)).map_err(io)?;
    // The count goes in the header after the last shader, as in the Python.
    output::pack_header(&mut pack, 0).map_err(io)?;

    let mut kinds: Vec<(String, usize)> = Vec::new();
    let mut failures: Vec<&str> = Vec::new();
    let mut seen: HashMap<&str, usize> = HashMap::new();
    let (mut count, mut total_words) = (0u32, 0usize);
    for (shader, d) in shaders.iter().zip(&decoded) {
        let (Some(fssm), Some(code)) = (&d.fssm, &d.code) else {
            failures.push(&shader.name);
            continue;
        };
        output::pack_record(&mut pack, &shader.name, &d.stage, code).map_err(io)?;
        let row = [
            shader.name.clone(),
            d.stage.clone(),
            format!("0x{:x}", shader.position()),
            d.blob.len().to_string(),
            fssm.len().to_string(),
            code.words.len().to_string(),
            code.padding.to_string(),
            output::hex(&code.sub_header),
        ];
        output::csv_row(&mut table, &row).map_err(io)?;
        match kinds.iter_mut().find(|(k, _)| *k == d.stage) {
            Some((_, n)) => *n += 1,
            None => kinds.push((d.stage.clone(), 1)),
        }
        total_words += code.words.len();
        count += 1;
        if args.files {
            let n = seen.entry(&shader.name).or_default();
            *n += 1;
            let stem = safe_file_name(&shader.name) + &if *n > 1 { format!("_{n}") } else { String::new() };
            let words: Vec<u8> = code.words.iter().flat_map(|w| w.to_le_bytes()).collect();
            fs::write(args.out.join("dxbc").join(format!("{stem}.dxbc")), &d.blob).map_err(io)?;
            fs::write(args.out.join("fssm").join(format!("{stem}.fssm")), fssm).map_err(io)?;
            fs::write(args.out.join("code").join(format!("{stem}.bin")), words).map_err(io)?;
        }
    }
    let mut pack = pack.into_inner().map_err(|e| e.to_string())?;
    pack.flush().map_err(io)?;
    use std::io::{Seek, SeekFrom};
    pack.seek(SeekFrom::Start(8)).map_err(io)?;
    pack.write_all(&count.to_le_bytes()).map_err(io)?;
    table.flush().map_err(io)?;

    report(shaders.len(), count, total_words, &kinds, &failures, args);
    Ok(u8::from(!failures.is_empty()))
}

/// A file name made of the characters of `text` that are safe in one.
fn safe_file_name(text: &str) -> String {
    text.chars().map(|c| if c.is_alphanumeric() || "._-".contains(c) { c } else { '_' }).take(150).collect()
}

fn report(found: usize, count: u32, total_words: usize, kinds: &[(String, usize)], failures: &[&str], args: &Args) {
    println!("Shaders trouves : {found}");
    println!("Code GPU isole : {count} ({total_words} mots, {:.1} Mo)", total_words as f64 * 4.0 / 1_048_576.0);
    // Most common first; equal counts keep their first appearance, as Counter.most_common does.
    let mut order: BTreeMap<std::cmp::Reverse<usize>, Vec<&str>> = BTreeMap::new();
    for (kind, n) in kinds {
        order.entry(std::cmp::Reverse(*n)).or_default().push(kind);
    }
    for (std::cmp::Reverse(n), names) in order {
        for kind in names {
            println!("  {kind:<10} {n}");
        }
    }
    let examples: Vec<String> = failures.iter().take(3).map(|f| format!("'{f}'")).collect();
    println!("Echecs : {}{}", failures.len(), if failures.is_empty() { String::new() } else { format!(" (ex. [{}])", examples.join(", ")) });
    println!("Sorties : {}", args.out.display());
}

fn main() -> ExitCode {
    match args().and_then(|a| run(&a)) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}
