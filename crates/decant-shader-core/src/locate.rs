//! `decant_shader_locate.py`: puts the inputs and outputs of a DecantShader SPIR-V module at the locations
//! Direct3D 12 (and vkd3d-proton) expects, the register number of each element in the shader
//! signature (ISG1, OSG1).
//!
//! The translator numbers them as the console hardware does:
//! * outputs of a vertex shader: parameter j is the non system output register of rank j
//! * inputs of a pixel shader: attribute k is the non system input register of rank k among those
//!   with a component read
//!
//! Inputs of a vertex shader already are at their register number.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;

use crate::meta::{self, REGISTER, USED};
use crate::py;
use crate::spirv::Table;

/// The non system output registers, in order: the vertex parameter j is register `[j]`.
pub fn output_registers(outputs: &[Value]) -> Result<Vec<i64>, String> {
    let mut registers = BTreeSet::new();
    for e in outputs {
        if meta::ordinary(e)? {
            registers.insert(meta::int(e, REGISTER)?);
        }
    }
    Ok(registers.into_iter().collect())
}

/// The non system input registers with at least one component read, in order: the pixel
/// attribute k is register `[k]`.
pub fn read_registers(inputs: &[Value]) -> Result<Vec<i64>, String> {
    let mut reads: BTreeMap<i64, i64> = BTreeMap::new();
    for e in inputs {
        if meta::ordinary(e)? {
            *reads.entry(meta::int(e, REGISTER)?).or_default() |= meta::int(e, USED)?;
        }
    }
    Ok(reads.into_iter().filter(|&(_, mask)| mask != 0).map(|(r, _)| r).collect())
}

/// Input and output tables for a shader, from its metadata row and the `outputs` of its outputs row.
pub fn tables(meta: &Value, outputs: &[Value]) -> Result<(Table, Table), String> {
    let stage = meta::field(meta, "stage")?;
    let numbered = |registers: Vec<i64>| registers.into_iter().enumerate().map(|(i, r)| (i as i64, r)).collect::<Table>();
    match stage.as_str() {
        Some("vertex") => Ok((Table::new(), numbered(output_registers(outputs)?))),
        Some("pixel") => Ok((numbered(read_registers(meta::list(meta, "inputs")?)?), Table::new())),
        _ => Err(format!("etage non gere : {}", py::str_value(stage))),
    }
}

/// [`tables`] for the shader `pack_id`, read from `out/decant_shader_metadata.jsonl` and
/// `out/decant_shader_outputs.jsonl` in `out`.
pub fn tables_for(out: &Path, pack_id: i64) -> Result<(Table, Table), String> {
    let meta = meta::row(&out.join("decant_shader_metadata.jsonl"), pack_id)?;
    let outputs = meta::row(&out.join("decant_shader_outputs.jsonl"), pack_id)?;
    tables(&meta, meta::list(&outputs, "outputs")?)
}

/// The line the tool prints: `entrees : {...} | sorties : {...}`.
pub fn report(inputs: &Table, outputs: &Table) -> String {
    format!("entrees : {} | sorties : {}", py::repr_int_dict(inputs), py::repr_int_dict(outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn vertex_outputs_go_to_their_register() {
        let meta = json!({"stage": "vertex", "inputs": []});
        let outputs = json!([
            ["SV_Position", 0, 0, 15, 15, 1, 3, 0, 0],
            ["TEXCOORD", 0, 3, 3, 3, 0, 3, 0, 0],
            ["TEXCOORD", 1, 3, 12, 12, 0, 3, 0, 0],
            ["COLOR", 0, 1, 15, 15, 0, 3, 0, 0]
        ]);
        let (inputs, outputs) = tables(&meta, outputs.as_array().unwrap()).unwrap();
        assert_eq!(report(&inputs, &outputs), "entrees : {} | sorties : {0: 1, 1: 3}");
    }

    #[test]
    fn pixel_inputs_skip_unread_registers() {
        let meta = json!({"stage": "pixel", "inputs": [
            ["SV_Position", 0, 0, 15, 15, 1, 3, 0, 0],
            ["COLOR", 0, 2, 15, 0, 0, 3, 0, 0],
            ["TEXCOORD", 0, 4, 3, 0, 0, 3, 0, 0],
            ["TEXCOORD", 1, 4, 12, 4, 0, 3, 0, 0],
            ["NORMAL", 0, 1, 7, 7, 0, 3, 0, 0]
        ]});
        let (inputs, outputs) = tables(&meta, &[]).unwrap();
        assert_eq!(report(&inputs, &outputs), "entrees : {0: 1, 1: 4} | sorties : {}");
    }

    #[test]
    fn other_stages_are_refused() {
        assert_eq!(tables(&json!({"stage": "compute"}), &[]).unwrap_err(), "etage non gere : compute");
        assert_eq!(tables(&json!({"stage": null}), &[]).unwrap_err(), "etage non gere : None");
    }
}
