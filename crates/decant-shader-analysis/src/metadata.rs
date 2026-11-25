//! `decant_shader_metadata.py`: the `PSV0` and `ISG1` reflection of every shader, and an independent
//! comparison with fields of the FSSM header. One JSON line per shader, then a summary line.

use decant_extract_shaders::output::stage_code;

use crate::dxbc::{self, Binding, Element, u16_at, u32_at, u64_at};
use crate::py::{Counter, Py};
use crate::tuple;

/// A step of the FSSM field correlation of a compute shader, in the order the script counts them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Hit {
    /// The word at `field` equals the thread count of `axis`.
    Axis { axis: usize, field: usize },
    /// The word at `field` was compared with the fixed wave size `wave`.
    Wave { wave: u32, field: usize, hit: bool },
}

/// What one shader contributes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shader {
    pub stage: String,
    pub wave_bounds: [u32; 2],
    pub threads: Option<[u32; 3]>,
    pub mesh_limits: [u16; 2],
    pub inputs: Vec<Element>,
    pub bindings: Vec<Binding>,
    pub fssm_kind: u32,
    pub heap_flags: u32,
    pub hits: Vec<Hit>,
}

impl Shader {
    /// `lo` when both wave bounds are 32 or both are 64.
    pub fn wave(&self) -> Option<u32> {
        let [lo, hi] = self.wave_bounds;
        (lo == hi && (lo == 32 || lo == 64)).then_some(lo)
    }

    /// The JSON line of the shader, without its line end.
    pub fn row(&self, id: usize, name: &str) -> String {
        let opt = |v: Option<u32>| v.map_or(Py::None, Py::from);
        Py::Dict(vec![
            ("id".into(), id.into()),
            ("name".into(), name.into()),
            ("stage".into(), self.stage.clone().into()),
            ("wave_bounds".into(), self.wave_bounds.to_vec().into()),
            ("wave".into(), opt(self.wave())),
            ("threads".into(), self.threads.map_or(Py::None, |t| t.to_vec().into())),
            ("mesh_limits".into(), self.mesh_limits.iter().map(|&v| u32::from(v)).collect::<Vec<_>>().into()),
            ("inputs".into(), Py::List(self.inputs.iter().map(Element::py).collect())),
            ("bindings".into(), Py::List(self.bindings.iter().map(Binding::py).collect())),
            ("fssm_kind".into(), self.fssm_kind.into()),
            ("heap_flags".into(), self.heap_flags.into()),
        ])
        .json(true)
    }
}

/// Reads one shader container, with the checks of the script in its order.
pub fn analyse(blob: &[u8], fssm: Option<&[u8]>) -> Result<Shader, String> {
    let blocks = dxbc::blocks(blob);
    let p = dxbc::block(&blocks, "PSV0")?;
    let (stage, _, _) = decant_extract_shaders::container::kind(blob);
    let ri = u32_at(p, 0)?;
    let code = stage_code(&stage);
    if code == 255 {
        return Err(format!("KeyError: '{stage}'"));
    }
    let stage_byte = *p.get(28).ok_or("IndexError: PSV0 shorter than 29 bytes")?;
    if ri < 48 || stage_byte != code {
        return Err(format!("AssertionError: PSV0 information size {ri}, stage byte {stage_byte} for {stage}"));
    }
    let (lo, hi) = (u32_at(p, 20)?, u32_at(p, 24)?);
    let threads = if stage == "mesh" || stage == "compute" { Some([u32_at(p, 40)?, u32_at(p, 44)?, u32_at(p, 48)?]) } else { None };
    let bindings = dxbc::psv0_bindings(p, ri)?;
    let inputs = dxbc::signature(dxbc::block(&blocks, "ISG1")?)?;
    let fs = fssm.ok_or("TypeError: the shader has no FSSM block")?;
    let (offset, size) = (u64_at(fs, 0xC0)?, u64_at(fs, 0xC8)?);
    if u128::from(offset) + 16 + u128::from(size) > fs.len() as u128 {
        return Err("AssertionError: FSSM code runs past the block".to_owned());
    }
    let fssm_kind = u32_at(fs, 16)?;
    let mut hits = Vec::new();
    if stage == "compute" {
        let fssm_threads = [u32_at(fs, 0x13C)?, u32_at(fs, 0x140)?, u32_at(fs, 0x144)?];
        if Some(fssm_threads) != threads {
            return Err(format!("AssertionError: FSSM threads {fssm_threads:?}, PSV0 threads {threads:?}"));
        }
        let dims = fssm_threads;
        let end = offset.min(0x140) as i64 - 3;
        for field in (0..end.max(0) as usize).step_by(4) {
            let val = u32_at(fs, field)?;
            for (axis, &d) in dims.iter().enumerate() {
                if val == d {
                    hits.push(Hit::Axis { axis, field });
                }
            }
            if lo == hi && (lo == 32 || lo == 64) {
                hits.push(Hit::Wave { wave: lo, field, hit: val == lo });
            }
        }
    }
    let mesh_limits = if stage == "mesh" { [u16_at(p, 16)?, u16_at(p, 18)?] } else { [0, 0] };
    let heap_flags = dxbc::heap_flags(dxbc::block(&blocks, "SFI0")?);
    Ok(Shader { stage, wave_bounds: [lo, hi], threads, mesh_limits, inputs, bindings, fssm_kind, heap_flags, hits })
}

/// The JSON line of `decant_shader_outputs.py` for one shader: its `OSG1` elements.
pub fn outputs_row(id: usize, name: &str, blob: &[u8]) -> Result<String, String> {
    let blocks = dxbc::blocks(blob);
    let outputs = dxbc::signature(dxbc::block(&blocks, "OSG1")?)?;
    let (stage, _, _) = decant_extract_shaders::container::kind(blob);
    Ok(Py::Dict(vec![
        ("id".into(), id.into()),
        ("name".into(), name.into()),
        ("stage".into(), stage.into()),
        ("outputs".into(), Py::List(outputs.iter().map(Element::py).collect())),
    ])
    .json(true))
}

/// The counters of the summary line.
#[derive(Default)]
pub struct Summary {
    shaders: usize,
    wave: Counter<(String, u32, u32)>,
    resources: Counter<(String, &'static str)>,
    fssm: Vec<(String, Counter<u32>)>,
    threads: Counter<(String, [u32; 3])>,
    matched: Vec<(usize, Counter<usize>)>,
    matches_wave: Vec<(u32, Counter<usize>)>,
}

impl From<(String, u32, u32)> for Py {
    fn from((a, b, c): (String, u32, u32)) -> Py {
        tuple!(a, b, c)
    }
}

impl From<(String, &'static str)> for Py {
    fn from((a, b): (String, &'static str)) -> Py {
        tuple!(a, b)
    }
}

impl From<(String, [u32; 3])> for Py {
    fn from((a, [x, y, z]): (String, [u32; 3])) -> Py {
        tuple!(a, tuple!(x, y, z))
    }
}

/// The counter stored under `key` in an insertion ordered `defaultdict(Counter)`.
fn slot<K: PartialEq, V: Default>(map: &mut Vec<(K, V)>, key: K) -> &mut V {
    let i = match map.iter().position(|(k, _)| *k == key) {
        Some(i) => i,
        None => {
            map.push((key, V::default()));
            map.len() - 1
        }
    };
    &mut map[i].1
}

impl Summary {
    pub fn add(&mut self, s: &Shader) {
        self.shaders += 1;
        self.wave.add((s.stage.clone(), s.wave_bounds[0], s.wave_bounds[1]), 1);
        for b in &s.bindings {
            self.resources.add((s.stage.clone(), b.prefix), 1);
        }
        slot(&mut self.fssm, s.stage.clone()).add(s.fssm_kind, 1);
        if let Some(dims) = s.threads {
            self.threads.add((s.stage.clone(), dims), 1);
        }
        for hit in &s.hits {
            match *hit {
                Hit::Axis { axis, field } => slot(&mut self.matched, axis).add(field, 1),
                Hit::Wave { wave, field, hit } => slot(&mut self.matches_wave, wave).add(field, i64::from(hit)),
            }
        }
    }

    /// The summary line, as the script's final `print`, without its line end.
    pub fn line(&self) -> String {
        let dict = |pairs: Vec<(Py, Py)>| Py::Dict(pairs).repr();
        let fssm = dict(self.fssm.iter().map(|(k, c)| (Py::from(k.clone()), Py::Raw(c.repr()))).collect());
        let matched = dict(self.matched.iter().map(|(k, c)| (Py::from(*k), c.most_common_py(4))).collect());
        let wave = dict(self.matches_wave.iter().map(|(k, c)| (Py::from(*k), c.most_common_py(4))).collect());
        [
            "shaders".to_owned(),
            self.shaders.to_string(),
            "wave".to_owned(),
            self.wave.repr(),
            "resources".to_owned(),
            self.resources.repr(),
            "fssm_kind".to_owned(),
            fssm,
            "top_threads".to_owned(),
            self.threads.most_common_py(15).repr(),
            "correlation size (compute)".to_owned(),
            matched,
            "wave candidates".to_owned(),
            wave,
        ]
        .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn script_assertions_become_errors() {
        let library = synthetic::Container::new(6).build();
        assert!(analyse(&library, decant_extract_shaders::container::gpu_code(&library).as_deref()).unwrap_err().starts_with("KeyError"));
        let mut wrong = synthetic::Container::new(0);
        wrong.psv_stage = 1;
        let blob = wrong.build();
        assert!(analyse(&blob, decant_extract_shaders::container::gpu_code(&blob).as_deref()).unwrap_err().starts_with("AssertionError"));
        let blob = synthetic::Container::new(0).without("SFI0").build();
        assert_eq!(analyse(&blob, decant_extract_shaders::container::gpu_code(&blob).as_deref()).unwrap_err(), "KeyError: 'SFI0'");
        let blob = synthetic::Container::new(0).build();
        assert!(analyse(&blob, None).unwrap_err().starts_with("TypeError"));
    }
}
