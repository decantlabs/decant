//! What the direct shader path asks of a translator (issue #172). A [`ShaderSource`] gets the
//! console bytes of every stage of one pipeline, the parsed root signature of that pipeline and the
//! linkage between its stages, and gives back per stage HLSL with its entry point and target
//! profile, or a refusal with a reason.
//!
//! The layer compiles the HLSL with DXC (`decant-dxc`), caches the DXIL on disk (`direct_cache`)
//! and makes the pipeline with it (`compile_service`, `direct_pso`). The translator itself
//! (`decant-shader`) is behind this trait in `direct_source`.
//!
//! [`FixedSource`] answers fixed HLSL for the tests.

use std::sync::Arc;

use crate::rts0::RootLayout;

/// A shader stage of a pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageKind {
    /// Vertex shader.
    Vertex,
    /// Pixel shader.
    Pixel,
    /// Domain shader.
    Domain,
    /// Hull shader.
    Hull,
    /// Geometry shader.
    Geometry,
    /// Compute shader.
    Compute,
    /// Amplification shader (mesh pipelines). The title has none: a translator refuses it.
    Amplification,
    /// Mesh shader (mesh pipelines). Its outputs are the pixel shader's inputs, so both come in one
    /// [`Pipeline`].
    Mesh,
}

impl StageKind {
    /// The short name used in log lines and profiles: `vs`, `ps`, `ds`, `hs`, `gs`, `cs`, `as`, `ms`.
    pub fn short(self) -> &'static str {
        match self {
            StageKind::Vertex => "vs",
            StageKind::Pixel => "ps",
            StageKind::Domain => "ds",
            StageKind::Hull => "hs",
            StageKind::Geometry => "gs",
            StageKind::Compute => "cs",
            StageKind::Amplification => "as",
            StageKind::Mesh => "ms",
        }
    }

    /// The `D3D12_SHADER_VISIBILITY` of the stage, 0 (all) for compute.
    pub fn visibility(self) -> u32 {
        match self {
            StageKind::Vertex => 1,
            StageKind::Hull => 2,
            StageKind::Domain => 3,
            StageKind::Geometry => 4,
            StageKind::Pixel => 5,
            StageKind::Compute => 0,
            StageKind::Amplification => 6,
            StageKind::Mesh => 7,
        }
    }
}

/// The console bytes of one stage, exactly as the title passed them to the pipeline creation.
#[derive(Clone, Debug)]
pub struct StageInput {
    /// Which stage.
    pub kind: StageKind,
    /// The console program bytes (shared, not copied, with the pipeline that holds them).
    pub bytes: Arc<[u8]>,
}

/// One element of the input layout of a graphics pipeline (`D3D12_INPUT_ELEMENT_DESC`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InputElement {
    /// Semantic name.
    pub semantic: String,
    /// Semantic index.
    pub index: u32,
    /// `DXGI_FORMAT`.
    pub format: u32,
    /// Input slot.
    pub slot: u32,
    /// Byte offset in the slot.
    pub offset: u32,
    /// `D3D12_INPUT_CLASSIFICATION`.
    pub per_instance: bool,
    /// Instance data step rate.
    pub step_rate: u32,
}

/// What links the stages of a pipeline to each other and to the fixed function parts around them.
/// Part of the cache key: two pipelines with the same console bytes but another linkage may need
/// other HLSL.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Linkage {
    /// The stages of the pipeline in the order the description has them (VS, PS, DS, HS, GS, or CS,
    /// or AS, MS, PS for a mesh pipeline).
    pub stages: Vec<StageKind>,
    /// The input layout (graphics only).
    pub inputs: Vec<InputElement>,
    /// The `DXGI_FORMAT` of each render target (graphics only).
    pub targets: Vec<u32>,
    /// The `DXGI_FORMAT` of the depth target, 0 when none.
    pub depth_format: u32,
    /// `D3D12_PRIMITIVE_TOPOLOGY_TYPE`.
    pub topology_type: u32,
    /// Sample count of the targets.
    pub sample_count: u32,
    /// Whether the pipeline has a stream output declaration.
    pub stream_output: bool,
}

/// Everything a translator gets for one pipeline.
#[derive(Clone, Debug)]
pub struct Pipeline {
    /// Every stage, in the order of [`Linkage::stages`].
    pub stages: Vec<StageInput>,
    /// The root signature the pipeline was created with, as the PC runtime got it: with the heap
    /// flags and the user data mirror the layer adds. `root.without_layer_params()` is the title's
    /// own model, and `root.mirror_index()` tells whether the mirror is there.
    pub root: RootLayout,
    /// How the stages link.
    pub linkage: Linkage,
}

/// HLSL for one stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageHlsl {
    /// The HLSL source.
    pub hlsl: String,
    /// The entry point (`main`).
    pub entry: String,
    /// The target profile (`vs_6_6`, `cs_6_6` and so on).
    pub profile: String,
    /// Extra DXC arguments (`-enable-16bit-types`, `-HV 2021`), each one item.
    pub args: Vec<String>,
}

/// Why a translator does not translate a stage or a pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// A short stable name of the reason, counted per run (`unknown opcode`, `mesh stage`).
    pub reason: String,
    /// Free text for the log (the opcode, the offset).
    pub detail: String,
}

impl Refusal {
    /// The reason of a pipeline whose translation is still running: not refused, not yet answered.
    /// The caller binds it as missing and asks again at the next bind.
    pub const PENDING: &'static str = "adapter:translation-pending";

    /// A refusal with `reason` and `detail`.
    pub fn new(reason: impl Into<String>, detail: impl Into<String>) -> Refusal {
        Refusal { reason: reason.into(), detail: detail.into() }
    }

    /// Whether this is no refusal but a translation that has not finished yet.
    pub fn is_pending(&self) -> bool {
        self.reason == Refusal::PENDING
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.detail.is_empty() {
            f.write_str(&self.reason)
        } else {
            write!(f, "{} ({})", self.reason, self.detail)
        }
    }
}

/// A translator from console programs to HLSL.
///
/// [`translate`] is called from the layer's compile workers. [`accepts`] and [`stage_keys`] run on
/// the title thread that first binds the pipeline, before anything is queued.
///
/// [`translate`]: ShaderSource::translate
/// [`accepts`]: ShaderSource::accepts
/// [`stage_keys`]: ShaderSource::stage_keys
pub trait ShaderSource: Send + Sync {
    /// Name and version of the translator. Part of the cache key: change it whenever the HLSL for
    /// the same input may change.
    fn version(&self) -> String;

    /// A look before any work is queued: a pipeline refused here is never queued and never bound as
    /// missing, it takes the title's shaders at once. The default accepts everything.
    fn accepts(&self, _pipeline: &Pipeline) -> Result<(), Refusal> {
        Ok(())
    }

    /// What decides the DXIL of each stage, one key per stage in the order of `pipeline.stages`,
    /// from a source that knows it better than the default: the console bytes of the stage, the
    /// root signature and the linkage. A source whose HLSL for a stage follows another stage too
    /// (a pixel stage linked to the vertex outputs) must answer here, with a key over the HLSL
    /// text for example. Asked after [`accepts`](ShaderSource::accepts) said yes.
    fn stage_keys(&self, _pipeline: &Pipeline) -> Option<Vec<u128>> {
        None
    }

    /// The HLSL of every stage of `pipeline`, one entry per stage in the order of
    /// `pipeline.stages`. The pipeline takes the direct path only when every stage translates.
    fn translate(&self, pipeline: &Pipeline) -> Vec<Result<StageHlsl, Refusal>>;
}

/// A translator for the tests: the same fixed HLSL for every stage of a kind it knows, a refusal
/// for the others. Stages whose bytes start with `REFUSE` are refused too.
#[derive(Clone, Debug, Default)]
pub struct FixedSource {
    /// HLSL, entry and profile per stage kind.
    pub stages: Vec<(StageKind, StageHlsl)>,
    /// Bumps the version (and so the cache key).
    pub generation: u32,
}

impl FixedSource {
    /// A compute shader that writes nothing, and a vertex and a pixel shader that draw a constant
    /// colour, all for shader model 6.0.
    pub fn simple() -> FixedSource {
        let stage = |hlsl: &str, profile: &str| StageHlsl { hlsl: hlsl.to_string(), entry: "main".to_string(), profile: profile.to_string(), args: Vec::new() };
        FixedSource {
            stages: vec![
                (StageKind::Compute, stage("[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {}\n", "cs_6_0")),
                (StageKind::Vertex, stage("float4 main(uint id : SV_VertexID) : SV_Position { return float4(float(id & 1), float(id >> 1), 0, 1); }\n", "vs_6_0")),
                (StageKind::Pixel, stage("float4 main() : SV_Target0 { return float4(1, 0, 1, 1); }\n", "ps_6_0")),
            ],
            generation: 0,
        }
    }
}

impl ShaderSource for FixedSource {
    fn version(&self) -> String {
        format!("fixed {}", self.generation)
    }

    fn translate(&self, pipeline: &Pipeline) -> Vec<Result<StageHlsl, Refusal>> {
        pipeline
            .stages
            .iter()
            .map(|s| {
                if s.bytes.starts_with(b"REFUSE") {
                    return Err(Refusal::new("test refusal", s.kind.short()));
                }
                self.stages.iter().find(|(k, _)| *k == s.kind).map(|(_, h)| h.clone()).ok_or_else(|| Refusal::new("stage not known", s.kind.short()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pipeline(stages: &[(StageKind, &[u8])]) -> Pipeline {
        Pipeline {
            stages: stages.iter().map(|(k, b)| StageInput { kind: *k, bytes: Arc::from(*b) }).collect(),
            root: RootLayout { version: 2, flags: 0, params: Vec::new(), static_samplers: Vec::new() },
            linkage: Linkage { stages: stages.iter().map(|(k, _)| *k).collect(), ..Linkage::default() },
        }
    }

    #[test]
    fn the_fixed_source_answers_per_stage_and_refuses_marked_bytes() {
        let s = FixedSource::simple();
        let out = s.translate(&pipeline(&[(StageKind::Vertex, b"vs"), (StageKind::Pixel, b"REFUSE ps"), (StageKind::Hull, b"hs")]));
        assert_eq!(out[0].as_ref().map(|h| h.profile.as_str()), Ok("vs_6_0"));
        assert_eq!(out[1].as_ref().map_err(|e| e.reason.as_str()), Err("test refusal"));
        assert_eq!(out[2].as_ref().map_err(|e| e.to_string()), Err("stage not known (hs)".to_string()));
    }
}
