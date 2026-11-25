//! The shader gallery of `harness/decant_shader_gallery.py`: runs thousands of translated pixel
//! shaders on the GPU and sorts the result. The parts that do not run anything live here.
//!
//! For each shader: translation and description (`decant_shader_describe`), reading of the module's
//! resources (`spirv_reflect`), then one run per process in `decant_shader_run --layout` on a full screen
//! triangle with the vertex shader VS_Blit (number 442). The bench makes every input itself.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use serde_json::Value;
use decant_shader_batch::process::{self, Ended};
use decant_shader_tools::reflect::{Operand, Reflection, reflect};

/// `VS_Blit`.
pub const VERTEX: i64 = 442;
/// The size the bench gives every buffer, written as each buffer's push constant size.
pub const BUFFER_BYTES: i64 = 65536;

/// `str(value)` for a decoration operand.
fn text(o: Operand) -> String {
    match o {
        Operand::Int(v) => v.to_string(),
        Operand::Flag => "True".into(),
        Operand::Unknown => "?".into(),
    }
}

/// `int(value)` for a decoration operand.
fn int(o: Operand) -> Result<i64, String> {
    match o {
        Operand::Int(v) => Ok(i64::from(v)),
        Operand::Flag => Ok(1),
        Operand::Unknown => Err("ValueError: invalid literal for int() with base 10: '?'".into()),
    }
}

/// `layout_lines()`: one line per resource for `decant_shader_run --layout`. The page table and the
/// fault buffer of modules that read memory by address keep a kind of their own, so the bench
/// fills them instead of zeroing them.
pub fn layout_lines(reflection: &Reflection) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    for r in &reflection.resources {
        let kind = match r.name.as_str() {
            "bda_pagetable" | "fault_buffer" => r.name.as_str(),
            _ => r.kind.as_str(),
        };
        let image = r.image.as_ref();
        let fields = [
            text(r.set),
            text(r.binding),
            kind.to_owned(),
            r.count.to_string(),
            image.map_or("-".into(), |i| i.dim.to_owned()),
            u8::from(image.is_some_and(|i| i.arrayed)).to_string(),
            image.map_or("-".into(), |i| i.sampled_type.to_owned()),
            image.map_or("0".into(), |i| text(i.format)),
            u8::from(image.is_some_and(|i| i.ms)).to_string(),
            image.map_or(Ok(0), |i| int(i.depth))?.to_string(),
        ];
        lines.push(fields.join(" "));
    }
    Ok(lines)
}

/// `push_words()`: the translator's push constants. Offsets stay 0, then one size per bound buffer from the
/// word the description gives, below word 32.
pub fn push_words(description: &Value) -> Result<Vec<(i64, i64)>, String> {
    let Some(push) = description.get("push_constants").filter(|p| decant_shader_core::py::truthy(p)) else {
        return Ok(Vec::new());
    };
    let resources = description.get("resources").and_then(Value::as_array).ok_or("KeyError: 'resources'")?;
    let mut count = 0;
    for r in resources {
        if r.get("kind").is_some_and(|k| k == "buffer")
            && let Some(rank) = r.get("array_rank").and_then(Value::as_i64)
        {
            count = count.max(rank + 1);
        }
    }
    let first = push.get("buffer_sizes_dword").and_then(Value::as_i64).ok_or("KeyError: 'buffer_sizes_dword'")?;
    Ok((0..count).filter(|i| first + i < 32).map(|i| (first + i, BUFFER_BYTES)).collect())
}

/// `classify_image()`: `nan`, `non_dessine` (every pixel kept the clear value -1), `zero` or `ok`.
pub fn classify_image(data: &[u8]) -> &'static str {
    let values: Vec<f32> = data.as_chunks::<4>().0.iter().map(|w| f32::from_le_bytes(*w)).collect();
    if values.iter().any(|v| !v.is_finite()) {
        return "nan";
    }
    let drawn: Vec<&[f32]> = values.chunks(4).filter(|p| *p != [-1.0, -1.0, -1.0, -1.0]).collect();
    if drawn.is_empty() {
        return "non_dessine";
    }
    if drawn.iter().all(|p| p.iter().all(|&v| v == 0.0)) {
        return "zero";
    }
    "ok"
}

/// `str(x)` for a float.
pub fn float(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 { format!("{x:.1}") } else { format!("{x}") }
}

/// `json.dumps(value, ensure_ascii=False)`: compact, with `", "` and `": "` separators.
pub fn dumps(value: &Value) -> String {
    decant_shader_bridge::describe::compact(value)
}

/// Everything `run_one()` needs besides the shader.
pub struct Bench<'a> {
    pub program: &'a Path,
    pub out: &'a Path,
    pub vertex_spv: &'a Path,
    pub timeout: f64,
    pub robust: bool,
}

/// `run_one()`: runs one translated pixel shader and gives its verdict and detail.
pub fn run_one(bench: &Bench, pack_id: i64, description: &Value) -> Result<(String, String), String> {
    let spv = bench.out.join("modules").join(format!("{pack_id}.spv"));
    let bytes = std::fs::read(&spv).map_err(|e| format!("{}: {e}", spv.display()))?;
    let reflection = reflect(&bytes)?;
    let layout = bench.out.join("layouts").join(format!("{pack_id}.txt"));
    let text = layout_lines(&reflection)?.join("\n") + "\n";
    std::fs::write(&layout, decant_shader_core::py::native_newlines(&text)).map_err(|e| format!("{}: {e}", layout.display()))?;
    let image = bench.out.join("images").join(format!("{pack_id}.f32"));
    let _ = std::fs::remove_file(&image);
    let mut command = Command::new(bench.program);
    command.arg(bench.vertex_spv).arg(&spv).arg(&image).arg("--layout").arg(&layout);
    if !bench.robust {
        command.arg("--no-robust");
    }
    for (word, value) in push_words(description)? {
        command.arg("--push").arg(word.to_string()).arg(value.to_string());
    }
    let ran = process::run(&mut command, Duration::from_secs_f64(bench.timeout), true).map_err(|e| format!("{}: {e}", bench.program.display()))?;
    let code = match ran.ended {
        Ended::Timeout => return Ok(("delai".into(), format!("plus de {} s", float(bench.timeout)))),
        Ended::Exit(code) => code,
    };
    let (stdout, stderr) = (decant_shader_core::py::read_text(&ran.stdout), decant_shader_core::py::read_text(&ran.stderr));
    let shown = if stderr.is_empty() { stdout } else { stderr };
    let message = decant_shader_bridge::py::splitlines(decant_shader_bridge::py::strip(&shown)).last().map_or(String::new(), |l| (*l).to_owned());
    if code == 5 {
        return Ok(("saute".into(), message));
    }
    if code != 0 {
        if message.contains("vkCreateGraphicsPipelines") || message.contains("vkCreateShaderModule") {
            return Ok(("pipeline".into(), message));
        }
        return Ok(("plantage".into(), format!("code {code} : {message}")));
    }
    let Ok(data) = std::fs::read(&image) else {
        return Ok(("plantage".into(), "pas d'image".into()));
    };
    let mut verdict = classify_image(&data);
    if verdict == "non_dessine" {
        let has_color = reflection.outputs.iter().any(|o| o.location == Some(Operand::Int(0)) && o.builtin.is_none());
        verdict = if has_color { "rejete" } else { "sans_couleur" };
    }
    Ok((verdict.into(), message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use decant_shader_tools::reflect::{Image, Kind, Resource};

    fn resource(name: &str, kind: Kind, image: Option<Image>) -> Resource {
        Resource { set: Operand::Int(0), binding: Operand::Int(3), count: 1, name: name.into(), kind, image }
    }

    #[test]
    fn layout_lines_follow_the_python() {
        let image = Image { dim: "2d", depth: Operand::Int(0), arrayed: true, ms: false, sampled_type: "float", format: Operand::Int(0) };
        let mut r = Reflection { version: 0x10500, capabilities: vec![], resources: vec![], push_size: 0, inputs: vec![], outputs: vec![] };
        r.resources.push(resource("", Kind::SampledImage, Some(image.clone())));
        r.resources.push(resource("bda_pagetable", Kind::StorageBuffer, None));
        r.resources.push(Resource { set: Operand::Flag, count: 0, ..resource("x", Kind::UniformBuffer, None) });
        assert_eq!(layout_lines(&r).unwrap(), ["0 3 sampled_image 1 2d 1 float 0 0 0", "0 3 bda_pagetable 1 - 0 - 0 0 0", "True 3 uniform_buffer 0 - 0 - 0 0 0"]);
        let unknown = Image { dim: "?", depth: Operand::Unknown, arrayed: false, ms: false, sampled_type: "?", format: Operand::Unknown };
        r.resources.push(resource("", Kind::CombinedImage, Some(unknown)));
        assert!(layout_lines(&r).is_err(), "int('?') fails in the Python too");
    }

    #[test]
    fn push_words_size_each_bound_buffer() {
        let d = json!({"push_constants": {"buffer_offsets_dword": 0, "buffer_sizes_dword": 30, "shader_data_dwords": null},
            "resources": [{"kind": "buffer", "array_rank": 2}, {"kind": "buffer", "array_rank": null}, {"kind": "image", "array_rank": 7}]});
        assert_eq!(push_words(&d).unwrap(), [(30, 65536), (31, 65536)]);
        assert_eq!(push_words(&json!({"push_constants": null})).unwrap(), []);
        assert_eq!(push_words(&json!({"push_constants": {"buffer_sizes_dword": 4}, "resources": []})).unwrap(), []);
    }

    #[test]
    fn images_are_classified_like_the_python() {
        let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        assert_eq!(classify_image(&bytes(&[-1.0; 8])), "non_dessine");
        assert_eq!(classify_image(&bytes(&[-1.0, -1.0, -1.0, -1.0, 0.0, -0.0, 0.0, 0.0])), "zero");
        assert_eq!(classify_image(&bytes(&[-1.0, -1.0, -1.0, -1.0, 0.5, 0.0, 0.0, 1.0])), "ok");
        assert_eq!(classify_image(&bytes(&[f32::NAN, 0.0, 0.0, 0.0])), "nan");
        // A cut last pixel is never the four value clear colour, so it counts as drawn.
        assert_eq!(classify_image(&bytes(&[-1.0, -1.0])), "ok");
    }

    #[test]
    fn rows_dump_like_json_dumps() {
        let row = json!({"pack_id": 3, "name": "PS_é\"x", "verdict": "ok", "detail": "a\tb"});
        assert_eq!(dumps(&row), "{\"pack_id\": 3, \"name\": \"PS_é\\\"x\", \"verdict\": \"ok\", \"detail\": \"a\\tb\"}");
        assert_eq!(dumps(&json!({"counts": {}, "timeout": 10.0, "robust": true, "x": [1, null]})), "{\"counts\": {}, \"timeout\": 10.0, \"robust\": true, \"x\": [1, null]}");
        assert_eq!((float(10.0), float(2.5)), ("10.0".to_owned(), "2.5".to_owned()));
    }
}
