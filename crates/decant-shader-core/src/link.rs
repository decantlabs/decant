//! `decant_shader_link.py`: links the outputs of a vertex shader to the inputs of a pixel shader by
//! semantic. On the console the driver makes this match.
//!
//! * vertex: the non system output of rank j (OSG1 registers in order) is exported as parameter j
//! * pixel: attribute k is the non system ISG1 input register of rank k among those with at least one
//!   component read. This holds for 97 % of a sample of 333. The others number differently and are
//!   not handled.
//!
//! Each `Location k` input of the pixel module then gets the location of the vertex parameter that
//! carries the same semantic.

use std::path::Path;

use serde_json::Value;

use crate::locate::{output_registers, read_registers};
use crate::meta::{self, REGISTER, SEMANTIC, SEMANTIC_INDEX, USED};
use crate::py;
use crate::spirv::Table;

/// Pixel attribute to vertex parameter, from the vertex `outputs` and the pixel `inputs`.
pub fn mapping(outputs: &[Value], inputs: &[Value]) -> Result<Table, String> {
    let out_registers = output_registers(outputs)?;
    // (semantic, index) to parameter. A later element with the same semantic replaces an earlier one.
    let mut parameter: Vec<((&Value, &Value), i64)> = Vec::new();
    for e in outputs {
        if meta::ordinary(e)? {
            let register = meta::int(e, REGISTER)?;
            let rank = out_registers.iter().position(|&r| r == register).expect("register listed above") as i64;
            let key = (meta::at(e, SEMANTIC)?, meta::at(e, SEMANTIC_INDEX)?);
            match parameter.iter_mut().find(|(k, _)| *k == key) {
                Some((_, p)) => *p = rank,
                None => parameter.push((key, rank)),
            }
        }
    }
    let mut result = Table::new();
    for (attribute, register) in read_registers(inputs)?.into_iter().enumerate() {
        let mut targets: Vec<Option<i64>> = Vec::new();
        for e in inputs {
            if meta::ordinary(e)? && meta::int(e, REGISTER)? == register && meta::int(e, USED)? != 0 {
                let key = (meta::at(e, SEMANTIC)?, meta::at(e, SEMANTIC_INDEX)?);
                targets.push(parameter.iter().find(|(k, _)| *k == key).map(|&(_, p)| p));
            }
        }
        let first = targets.first().copied().flatten();
        let unique = first.is_some() && targets.iter().all(|&t| t == first);
        if !unique {
            return Err(format!(
                "entree {attribute} (registre {register}) : pas de sortie de sommets unique pour {}",
                py::repr_set(&targets)
            ));
        }
        result.insert(attribute as i64, first.expect("checked above"));
    }
    Ok(result)
}

/// [`mapping`] for the vertex shader `vertex_id` and the pixel shader `pixel_id`, read from
/// `out/decant_shader_outputs.jsonl` and `out/decant_shader_metadata.jsonl` in `out`.
pub fn mapping_for(out: &Path, vertex_id: i64, pixel_id: i64) -> Result<Table, String> {
    let outputs = meta::row(&out.join("decant_shader_outputs.jsonl"), vertex_id)?;
    let outputs = meta::list(&outputs, "outputs")?;
    let inputs = meta::row(&out.join("decant_shader_metadata.jsonl"), pixel_id)?;
    mapping(outputs, meta::list(&inputs, "inputs")?)
}

/// The line the tool prints.
pub fn report(table: &Table) -> String {
    format!("attribut de pixels -> parametre de sommets : {}", py::repr_int_dict(table))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vertex() -> Value {
        json!([
            ["SV_Position", 0, 0, 15, 15, 1, 3, 0, 0],
            ["COLOR", 0, 1, 15, 15, 0, 3, 0, 0],
            ["TEXCOORD", 0, 4, 3, 3, 0, 3, 0, 0],
            ["TEXCOORD", 1, 4, 12, 12, 0, 3, 0, 0],
            ["NORMAL", 0, 6, 7, 7, 0, 3, 0, 0]
        ])
    }

    #[test]
    fn attributes_follow_the_semantic() {
        let pixel = json!([
            ["SV_Position", 0, 0, 15, 15, 1, 3, 0, 0],
            ["NORMAL", 0, 1, 7, 7, 0, 3, 0, 0],
            ["TEXCOORD", 1, 2, 3, 3, 0, 3, 0, 0],
            ["COLOR", 0, 3, 15, 0, 0, 3, 0, 0]
        ]);
        let table = mapping(vertex().as_array().unwrap(), pixel.as_array().unwrap()).unwrap();
        assert_eq!(report(&table), "attribut de pixels -> parametre de sommets : {0: 2, 1: 1}");
    }

    #[test]
    fn ambiguous_or_missing_semantics_are_refused() {
        let missing = json!([["BINORMAL", 0, 1, 7, 7, 0, 3, 0, 0]]);
        assert_eq!(
            mapping(vertex().as_array().unwrap(), missing.as_array().unwrap()).unwrap_err(),
            "entree 0 (registre 1) : pas de sortie de sommets unique pour {None}"
        );
        let split = json!([["COLOR", 0, 2, 3, 3, 0, 3, 0, 0], ["NORMAL", 0, 2, 12, 12, 0, 3, 0, 0], ["X", 0, 2, 12, 12, 0, 3, 0, 0]]);
        assert_eq!(
            mapping(vertex().as_array().unwrap(), split.as_array().unwrap()).unwrap_err(),
            "entree 0 (registre 2) : pas de sortie de sommets unique pour {0, 2, None}"
        );
    }
}
