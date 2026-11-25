//! Rewrites the `Location` decorations of a SPIR-V module, word by word, as `relocate()` in
//! `decant_shader_locate.py` and `decant_shader_link.py` does. Nothing else in the module changes.

use std::collections::{BTreeMap, BTreeSet};

/// A location table: old location to new location.
pub type Table = BTreeMap<i64, i64>;

const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const LOCATION: u32 = 30;
const INPUT: u32 = 1;
const OUTPUT: u32 = 3;
/// Words of the module header, before the first instruction.
const HEADER: usize = 5;

fn words(module: &[u8]) -> Result<Vec<u32>, String> {
    if !module.len().is_multiple_of(4) {
        return Err(format!("struct.error: unpack requires a buffer of {} bytes", module.len() / 4 * 4));
    }
    Ok(module.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect())
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn word(words: &[u32], i: usize) -> Result<u32, String> {
    words.get(i).copied().ok_or_else(|| "IndexError: list index out of range".to_owned())
}

/// Calls `visit` with the start of every instruction after the header. An instruction of zero
/// words would never end (the Python loops forever on it), so it is an error here.
fn each(words: &mut [u32], mut visit: impl FnMut(&mut [u32], usize) -> Result<(), String>) -> Result<(), String> {
    let mut i = HEADER;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        visit(words, i)?;
        if count == 0 {
            return Err(format!("instruction of zero words at word {i}"));
        }
        i += count;
    }
    Ok(())
}

fn store(value: i64) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| "struct.error: argument out of range".to_owned())
}

/// `decant_shader_locate.py`: each `Location` of an `Input` variable goes through `inputs`, of an
/// `Output` variable through `outputs`. Locations missing from a table stay as they are.
pub fn locate(module: &[u8], inputs: &Table, outputs: &Table) -> Result<Vec<u8>, String> {
    let mut words = words(module)?;
    let mut storage = BTreeMap::new();
    each(&mut words, |w, i| {
        if w[i] & 0xFFFF == OP_VARIABLE {
            let id = word(w, i + 2)?;
            storage.insert(id, word(w, i + 3)?);
        }
        Ok(())
    })?;
    let none = Table::new();
    each(&mut words, |w, i| {
        if w[i] & 0xFFFF == OP_DECORATE && word(w, i + 2)? == LOCATION {
            let table = match storage.get(&word(w, i + 1)?) {
                Some(&INPUT) => inputs,
                Some(&OUTPUT) => outputs,
                _ => &none,
            };
            let old = word(w, i + 3)?;
            w[i + 3] = store(table.get(&i64::from(old)).copied().unwrap_or(i64::from(old)))?;
        }
        Ok(())
    })?;
    Ok(bytes(&words))
}

/// `decant_shader_link.py`: each `Location` of a variable declared `Input` goes through `table`, which
/// must hold it.
pub fn link(module: &[u8], table: &Table) -> Result<Vec<u8>, String> {
    let mut words = words(module)?;
    let mut inputs = BTreeSet::new();
    each(&mut words, |w, i| {
        if w[i] & 0xFFFF == OP_VARIABLE && word(w, i + 3)? == INPUT {
            inputs.insert(word(w, i + 2)?);
        }
        Ok(())
    })?;
    each(&mut words, |w, i| {
        if w[i] & 0xFFFF == OP_DECORATE && word(w, i + 2)? == LOCATION && inputs.contains(&word(w, i + 1)?) {
            let old = i64::from(word(w, i + 3)?);
            w[i + 3] = store(*table.get(&old).ok_or_else(|| format!("KeyError: {old}"))?)?;
        }
        Ok(())
    })?;
    Ok(bytes(&words))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn op(code: u32, operands: &[u32]) -> Vec<u32> {
        let mut out = vec![((operands.len() as u32 + 1) << 16) | code];
        out.extend_from_slice(operands);
        out
    }

    /// An invented module: header, two decorations per variable, an input (id 10, location 0),
    /// an input (id 11, location 1), an output (id 12, location 0) and a private one (id 13).
    pub fn module() -> Vec<u8> {
        let mut w = vec![0x0723_0203, 0x0001_0500, 0, 20, 0];
        w.extend(op(17, &[1])); // OpCapability Shader
        w.extend(op(OP_DECORATE, &[10, LOCATION, 0]));
        w.extend(op(OP_DECORATE, &[11, LOCATION, 1]));
        w.extend(op(OP_DECORATE, &[12, LOCATION, 0]));
        w.extend(op(OP_DECORATE, &[13, LOCATION, 5]));
        w.extend(op(OP_DECORATE, &[10, 11, 2])); // BuiltIn, untouched
        w.extend(op(OP_VARIABLE, &[2, 10, INPUT]));
        w.extend(op(OP_VARIABLE, &[2, 11, INPUT]));
        w.extend(op(OP_VARIABLE, &[3, 12, OUTPUT]));
        w.extend(op(OP_VARIABLE, &[4, 13, 6]));
        bytes(&w)
    }

    fn locations(module: &[u8]) -> Vec<u32> {
        let w = words(module).unwrap();
        [10, 14, 18, 22].iter().map(|&i| w[i]).collect()
    }

    #[test]
    fn locate_rewrites_by_storage_class() {
        let inputs = Table::from([(0, 4), (1, 6)]);
        let outputs = Table::from([(0, 2)]);
        let out = locate(&module(), &inputs, &outputs).unwrap();
        assert_eq!(locations(&out), [4, 6, 2, 5]);
        assert_eq!(out.len(), module().len());
        assert_eq!(locations(&locate(&module(), &Table::new(), &Table::from([(5, 9)])).unwrap()), [0, 1, 0, 5]);
    }

    #[test]
    fn link_rewrites_inputs_only_and_needs_every_location() {
        let out = link(&module(), &Table::from([(0, 3), (1, 0)])).unwrap();
        assert_eq!(locations(&out), [3, 0, 0, 5]);
        assert_eq!(link(&module(), &Table::from([(0, 3)])).unwrap_err(), "KeyError: 1");
    }

    #[test]
    fn malformed_modules_are_errors() {
        assert!(locate(&[0; 22], &Table::new(), &Table::new()).is_err());
        let mut w = words(&module()).unwrap();
        w[5] = 0;
        assert!(link(&bytes(&w), &Table::new()).is_err());
        let short = bytes(&[0, 0, 0, 0, 0, (4 << 16) | OP_VARIABLE, 1]);
        assert!(locate(&short, &Table::new(), &Table::new()).is_err());
        assert_eq!(locate(&[0; 12], &Table::new(), &Table::new()).unwrap(), [0; 12]);
    }
}
