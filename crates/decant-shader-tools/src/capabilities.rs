//! `decant_shader_capabilities.py`: counts, in the SPIR-V modules of a measurement, those that rely on
//! an emulator mechanism (physical addresses, that is console memory read by address, image
//! tables and so on), and the most frequent bindings.

use std::fmt::Write;
use std::path::Path;

use decant_shader_analysis::NL;
use decant_shader_analysis::py::{Counter, Py};

use crate::files::{batches, modules, words};
use crate::pyfmt::IntSet;

/// The capabilities the script names.
pub fn name(capability: u64) -> &'static str {
    match capability {
        5347 => "PhysicalStorageBufferAddresses (memoire par adresse)",
        5301 => "RuntimeDescriptorArray",
        4427 => "Int64/autres",
        _ => "",
    }
}

/// The capabilities and the `Binding` decorations of one module, read up to the first
/// `OpFunction`, as sets in Python's order.
pub fn scan(words: &[u32]) -> Result<(IntSet, IntSet), String> {
    let (mut caps, mut binds) = (IntSet::default(), IntSet::default());
    let at = |k: usize| words.get(k).copied().map(u64::from).ok_or_else(|| "IndexError: tuple index out of range".to_owned());
    let mut i = 5;
    while i < words.len() {
        let (op, count) = (words[i] & 0xFFFF, (words[i] >> 16) as usize);
        if count == 0 {
            break;
        }
        if op == 17 {
            caps.insert(at(i + 1)?);
        } else if op == 71 && at(i + 2)? == 33 {
            binds.insert(at(i + 3)?);
        } else if op == 54 {
            break;
        }
        i += count;
    }
    Ok((caps, binds))
}

/// The counts of one measurement folder.
#[derive(Default)]
pub struct Tally {
    pub modules: usize,
    pub capabilities: Counter<u64>,
    pub bindings: Counter<u64>,
}

impl Tally {
    pub fn add(&mut self, words: &[u32]) -> Result<(), String> {
        self.modules += 1;
        let (caps, binds) = scan(words)?;
        for c in caps.iter() {
            self.capabilities.add(c, 1);
        }
        for b in binds.iter() {
            self.bindings.add(b, 1);
        }
        Ok(())
    }

    /// The lines printed for the folder named `folder`.
    pub fn report(&self, folder: &str) -> String {
        let mut out = format!("{folder}: {} modules{NL}", self.modules);
        for (c, k) in self.capabilities.most_common(None) {
            let share = 100.0 * k as f64 / self.modules as f64;
            write!(out, "   capacite {c:5} {:52} {k:6} ({share:.1} %){NL}", name(c)).unwrap();
        }
        let common = Py::List(self.bindings.most_common(Some(14)).into_iter().map(|(b, n)| Py::Tuple(vec![Py::Int(b as i64), Py::Int(n)])).collect());
        write!(out, "   bindings les plus frequents : {}{NL}", common.repr()).unwrap();
        out
    }
}

/// The whole report of one folder: every `*.spv` of every batch folder, batches in sorted order.
pub fn folder(folder: &str) -> Result<String, String> {
    let mut tally = Tally::default();
    for batch in batches(Path::new(folder))? {
        for module in modules(&batch)? {
            tally.add(&words(&module)?)?;
        }
    }
    Ok(tally.report(folder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::Module;

    #[test]
    fn capabilities_and_bindings_are_counted_once_per_module() {
        let mut tally = Tally::default();
        let a = Module::new().capability(1).capability(5347).capability(1).binding(10, 3).binding(11, 3).function().capability(4427);
        let b = Module::new().capability(1).capability(5301).binding(12, 0);
        tally.add(&a.words()).unwrap();
        tally.add(&b.words()).unwrap();
        tally.add(&Module::new().words()).unwrap();
        let report = tally.report("m");
        let expected = [
            "m: 3 modules".to_owned(),
            format!("   capacite     1 {:52}      2 (66.7 %)", ""),
            format!("   capacite  5347 {:52}      1 (33.3 %)", "PhysicalStorageBufferAddresses (memoire par adresse)"),
            format!("   capacite  5301 {:52}      1 (33.3 %)", "RuntimeDescriptorArray"),
            "   bindings les plus frequents : [(3, 1), (0, 1)]".to_owned(),
        ];
        assert_eq!(report, expected.map(|l| format!("{l}{NL}")).concat());
    }
}
