//! Port of `src/shaders/fssm_tables.py`: prints the tables placed after the code in the FSSM block
//! of the given shaders. Fields 0xE8 and 0xF0 of the FSSM header point to two tables, 0xF8 and 0xFC
//! count their entries. Reads the archive named by `DECANT_SHADERS_RPF` and
//! `out/decant_shader_metadata.jsonl`.
//!
//! Usage: `fssm_tables <shader number> [...] [--out folder]`

use std::collections::HashMap;
use std::process::ExitCode;

use decant_shader_analysis::tables::{binding_pairs, dump};
use decant_shader_analysis::{args, int_arg, load_fssm, open_corpus, read_jsonl, run};

fn main() -> ExitCode {
    run(|| {
        let (numbers, paths) = args()?;
        let items = open_corpus()?;
        let mut meta = HashMap::new();
        for m in read_jsonl(&paths.metadata())? {
            meta.insert(m.get("id").and_then(|v| v.as_i64()).ok_or("KeyError: 'id'")?, m);
        }
        for text in &numbers {
            let n = int_arg(text)?;
            // A negative number counts from the end, as a Python list index does.
            let index = if n < 0 { n + items.len() as i64 } else { n };
            let item = usize::try_from(index).ok().and_then(|i| items.get(i)).ok_or("IndexError: list index out of range")?;
            let fs = load_fssm(item)?;
            let pairs = binding_pairs(meta.get(&n).ok_or_else(|| format!("KeyError: {n}"))?)?;
            let mut out = String::new();
            let result = dump(&mut out, &fs, n, &item.name, &pairs);
            print!("{out}");
            result?;
        }
        Ok(())
    })
}
