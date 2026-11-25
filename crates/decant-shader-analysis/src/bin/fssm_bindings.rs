//! Port of `src/shaders/fssm_bindings.py`: the binding table of a console shader (FSSM table B,
//! pointed to by field 0xF0): which Direct3D register (b, t, s, u) sits at which offset of which
//! descriptor table.
//!
//! A record is five words: `[kind, offset in the table in 32 bit words, register, extra, flags]`
//! with `flags = (class << 16) | (scalar register holding the table address << 8) | low byte`.
//! Kinds seen: 0x300 texture (t), 0x302 constant buffer (b), 0x401 sampler (s).
//!
//! Usage:
//! * `fssm_bindings <number> [...]` prints the decoded table.
//! * `fssm_bindings --verifie <n> <tmp>` compares, on n shaders drawn at random (seed 41, as in
//!   Python), the table loads of the machine code with the table entries. The machine code comes
//!   from the decoder test bench, run as a separate program on `out/shaders_gpu.pack`.
//!
//! Options: `--out folder`, `--decoder program`. The shaders come from `DECANT_SHADERS_RPF`.

use std::process::ExitCode;

use decant_shader_analysis::decoder::Decoder;
use decant_shader_analysis::tables::{record_line, records, verify};
use decant_shader_analysis::{NL, args, int_arg, load_fssm, open_corpus, run};

fn main() -> ExitCode {
    run(|| {
        let (positional, paths) = args()?;
        let items = open_corpus()?;
        let missing = || "IndexError: list index out of range".to_owned();
        let first = positional.first().ok_or_else(missing)?;
        if first != "--verifie" {
            for text in &positional {
                let n = int_arg(text)?;
                // A negative number counts from the end, as a Python list index does.
                let index = if n < 0 { n + items.len() as i64 } else { n };
                let item = usize::try_from(index).ok().and_then(|i| items.get(i)).ok_or_else(missing)?;
                print!("== {n} {}{NL}", item.name);
                for r in records(&load_fssm(item)?)? {
                    print!("{}{NL}", record_line(&r));
                }
            }
            return Ok(());
        }
        let count = int_arg(positional.get(1).ok_or_else(missing)?)?;
        let tmp = std::path::PathBuf::from(positional.get(2).ok_or_else(missing)?);
        std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let names: Vec<&str> = items.iter().map(|i| i.name.as_str()).collect();
        let decoder = Decoder { exe: paths.decoder.clone(), pack: paths.pack() };
        let report = verify(&names, count, &|n| records(&load_fssm(&items[n])?), &mut |name| decoder.listing(&tmp, name))?;
        print!("{report}");
        Ok(())
    })
}
