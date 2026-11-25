use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn sources(dir: &Path, paths: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            sources(&entry.path(), paths)?;
        } else {
            paths.push(entry.path());
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let translator = Path::new("../decant-shader");
    println!(
        "cargo:rerun-if-changed={}",
        translator.join("src").display()
    );

    let mut paths = vec![
        translator.join("Cargo.toml"),
        PathBuf::from("src/direct_source.rs"),
    ];
    sources(&translator.join("src"), &mut paths)?;
    paths.sort_unstable();

    let mut hash = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58du128;
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read(&path)?;
        let name = path
            .to_str()
            .expect("UTF-8 translator path")
            .replace('\\', "/");
        // Normalize checkout line endings so Windows and Linux use the same key.
        let source = String::from_utf8(source)
            .map_err(io::Error::other)?
            .replace("\r\n", "\n");
        for field in [name.as_bytes(), source.as_bytes()] {
            for byte in (field.len() as u64)
                .to_le_bytes()
                .into_iter()
                .chain(field.iter().copied())
            {
                hash ^= u128::from(byte);
                hash = hash.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
            }
        }
    }
    println!("cargo:rustc-env=DECANT_DIRECT_TRANSLATOR_HASH={hash:032x}");
    Ok(())
}
