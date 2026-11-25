//! The census at pipeline level: the programs of one pipeline are translated together, so the
//! link between the stages (the pixel input struct built from the vertex outputs) is exercised.
//!
//! A pipeline list is any text that has the stages of a pipeline as a tag and a container hash:
//! `VS 28d887aa04ffa15a PS b40d9ca24fe3d23d`. The layer's log lines about created and refused
//! pipelines have that form, so a log can be given as it is. Tags: `VS HS DS GS PS CS MS AS`.
//! Equal pipelines are counted once and remember how often they were seen.
//!
//! Version 1 manifests also supply the root layout and fixed-function linkage. The translator
//! links the stages from their program interfaces. The fixed-function fields distinguish records
//! with otherwise identical shaders and root layouts.

use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::path::Path;

use super::Tally;
use crate::api::{
    CountingHost, MirrorAbi, Options, RegisterClass, RootParam, RootParamKind, RootRange,
    RootSignature, Stage, StageProgram, StaticSampler,
};

/// One pipeline: its stages in the order of the source, as `(tag, container hash)`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pipeline {
    pub stages: Vec<(String, u64)>,
    manifest: Option<ManifestPipeline>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ManifestPipeline {
    root: Option<u64>,
    linkage: Vec<String>,
}

impl Pipeline {
    /// The name in reports: `VS:28d887aa04ffa15a+PS:b40d9ca24fe3d23d`.
    pub fn id(&self) -> String {
        let stages = self
            .stages
            .iter()
            .map(|(t, h)| format!("{t}:{h:016x}"))
            .collect::<Vec<_>>()
            .join("+");
        match &self.manifest {
            None => stages,
            Some(p) => {
                let root = p.root.map_or_else(|| "-".into(), |id| format!("{id:016x}"));
                format!("{stages} root={root} {}", p.linkage.join(" "))
                    .trim_end()
                    .to_owned()
            }
        }
    }
}

const TAGS: [&str; 8] = ["VS", "HS", "DS", "GS", "PS", "CS", "MS", "AS"];

/// The stages named on one line: every tag directly followed by a 16 digit hex hash.
pub fn parse_line(line: &str) -> Option<Pipeline> {
    let toks: Vec<&str> = line
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | ':' | '(' | ')' | '[' | ']' | '='))
        .filter(|t| !t.is_empty())
        .collect();
    let mut stages = Vec::new();
    for w in toks.windows(2) {
        if TAGS.contains(&w[0])
            && w[1].len() == 16
            && let Ok(h) = u64::from_str_radix(w[1], 16)
        {
            stages.push((w[0].to_owned(), h));
        }
    }
    (!stages.is_empty()).then_some(Pipeline {
        stages,
        manifest: None,
    })
}

#[derive(Debug)]
struct ManifestRoot {
    signature: RootSignature,
    options: Options,
}

fn numbers<const N: usize>(text: &str) -> Option<[u32; N]> {
    let mut fields = text.split('.');
    let mut out = [0; N];
    for value in &mut out {
        *value = fields.next()?.parse().ok()?;
    }
    fields.next().is_none().then_some(out)
}

fn parse_root(body: &str) -> Option<ManifestRoot> {
    let mut fields = body.split_whitespace();
    let _version = fields.next()?.parse::<u32>().ok()?;
    let mut signature = RootSignature {
        flags: u32::from_str_radix(fields.next()?.strip_prefix("0x")?, 16).ok()?,
        ..RootSignature::default()
    };
    for field in fields {
        let (kind, rest) = field.split_once(':')?;
        let (visibility, data) = rest.split_once(':')?;
        let visibility = visibility.parse().ok()?;
        let kind = match kind {
            "table" => RootParamKind::Table(
                data.split('+')
                    .filter(|r| !r.is_empty())
                    .map(|r| {
                        let (class, values) = r.split_once('.')?;
                        let class = match class {
                            "b" => RegisterClass::Cbv,
                            "t" => RegisterClass::Srv,
                            "u" => RegisterClass::Uav,
                            "s" => RegisterClass::Sampler,
                            _ => return None,
                        };
                        let [base_register, space, count, offset, _flags] = numbers(values)?;
                        Some(RootRange {
                            class,
                            base_register,
                            space,
                            count,
                            offset,
                        })
                    })
                    .collect::<Option<_>>()?,
            ),
            "const" => {
                let [register, space, count] = numbers(data)?;
                RootParamKind::Constants {
                    register,
                    space,
                    count,
                }
            }
            "cbv" | "srv" | "uav" => {
                let [register, space] = numbers(data)?;
                let class = match kind {
                    "cbv" => RegisterClass::Cbv,
                    "srv" => RegisterClass::Srv,
                    _ => RegisterClass::Uav,
                };
                RootParamKind::Descriptor {
                    class,
                    register,
                    space,
                }
            }
            "sampler" => {
                let [register, space] = numbers(data)?;
                signature.static_samplers.push(StaticSampler {
                    register,
                    space,
                    visibility,
                });
                continue;
            }
            _ => return None,
        };
        if !signature.static_samplers.is_empty() {
            return None;
        }
        signature.params.push(RootParam { kind, visibility });
    }
    // Match the layer's all-stage root views and remove its appended parameters.
    let has_view = |class, register| {
        signature.params.iter().any(|p| {
            p.visibility == 0
                && p.kind
                    == RootParamKind::Descriptor {
                        class,
                        register,
                        space: 15,
                    }
        })
    };
    let mirror = has_view(RegisterClass::Cbv, 1).then(|| MirrorAbi {
        heap_map: has_view(RegisterClass::Srv, 0).then_some(0),
        view_shadow: has_view(RegisterClass::Srv, 1).then_some(1),
        ..MirrorAbi::default()
    });
    while signature.params.last().is_some_and(|p| {
        p.visibility == 0 && matches!(p.kind, RootParamKind::Descriptor { space: 13..=15, .. })
    }) {
        signature.params.pop();
    }
    Some(ManifestRoot {
        signature,
        options: Options {
            mirror,
            ..Options::default()
        },
    })
}

fn parse_pipeline(body: &str) -> Option<Pipeline> {
    let mut fields = body.split_whitespace();
    let root = match fields.next()? {
        "-" => None,
        id => Some(u64::from_str_radix(id, 16).ok()?),
    };
    let (mut linkage, mut stages) = (Vec::new(), Vec::new());
    while let Some(field) = fields.next() {
        if field.contains('=') {
            linkage.push(field.to_owned());
        } else {
            let hash = fields.next()?;
            if !TAGS.contains(&field) || hash.len() != 16 {
                return None;
            }
            stages.push((field.to_owned(), u64::from_str_radix(hash, 16).ok()?));
        }
    }
    linkage.sort();
    (!stages.is_empty()).then_some(Pipeline {
        stages,
        manifest: Some(ManifestPipeline { root, linkage }),
    })
}

#[derive(Debug, Default)]
pub struct PipelineLists {
    pub seen: BTreeMap<Pipeline, u64>,
    roots: HashMap<u64, ManifestRoot>,
}

impl PipelineLists {
    /// Adds a manifest or a layer log, streaming lines from disk.
    pub fn read(&mut self, path: &Path) -> Result<(), String> {
        let at = |e| format!("{}: {e}", path.display());
        let file = std::fs::File::open(path).map_err(|e| at(e.to_string()))?;
        let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
        let mut bytes = Vec::new();
        let mut line_number = 0;
        loop {
            bytes.clear();
            if reader
                .read_until(b'\n', &mut bytes)
                .map_err(|e| at(e.to_string()))?
                == 0
            {
                break;
            }
            line_number += 1;
            self.add_line(String::from_utf8_lossy(&bytes).trim_end())
                .map_err(|e| at(format!("line {line_number}: {e}")))?;
        }
        Ok(())
    }

    fn add_line(&mut self, line: &str) -> Result<(), &'static str> {
        let pipeline = if let Some(body) = line.strip_prefix("R ") {
            let (id, body) = body.split_once(' ').ok_or("missing root signature body")?;
            let id = u64::from_str_radix(id, 16).map_err(|_| "invalid root id")?;
            if crate::container::fnv1a64(body.as_bytes()) != id {
                return Err("root id does not match its text");
            }
            self.roots
                .insert(id, parse_root(body).ok_or("invalid root signature")?);
            return Ok(());
        } else if let Some(body) = line.strip_prefix("P ") {
            Some(parse_pipeline(body).ok_or("invalid pipeline record")?)
        } else if !line.starts_with('#') {
            parse_line(line)
        } else {
            None
        };
        if let Some(pipeline) = pipeline {
            *self.seen.entry(pipeline).or_default() += 1;
        }
        Ok(())
    }
}

/// What the translation of a set of pipelines gave.
#[derive(Debug, Default)]
pub struct PipelineCensus {
    pub pipelines: u64,
    pub translated: u64,
    /// Times seen, summed over all pipelines and over the translated ones.
    pub seen: u64,
    pub seen_translated: u64,
    /// Pipelines by the key of the refusal that stops them.
    pub blockers: BTreeMap<String, Tally>,
    /// Per pipeline id: `translated` or the refusal key.
    pub outcomes: BTreeMap<String, String>,
}

impl PipelineCensus {
    fn add(&mut self, order: usize, p: &Pipeline, seen: u64, key: Option<String>) {
        self.pipelines += 1;
        self.seen += seen;
        let id = p.id();
        match key {
            None => {
                self.translated += 1;
                self.seen_translated += seen;
                self.outcomes.insert(id, "translated".to_owned());
            }
            Some(key) => {
                self.blockers
                    .entry(key.clone())
                    .or_default()
                    .add(&id, order);
                self.outcomes.insert(id, key);
            }
        }
    }

    pub fn merge(&mut self, o: PipelineCensus) {
        self.pipelines += o.pipelines;
        self.translated += o.translated;
        self.seen += o.seen;
        self.seen_translated += o.seen_translated;
        self.outcomes.extend(o.outcomes);
        super::merge_tallies(&mut self.blockers, o.blockers);
    }

    /// The report text.
    pub fn report(&self, top: usize) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let w = &mut out;
        let pct = |a: u64, b: u64| {
            if b == 0 {
                0.0
            } else {
                a as f64 * 100.0 / b as f64
            }
        };
        writeln!(
            w,
            "pipelines: {} distinct, {} translated ({:.1}%), seen {} times of which {} translated ({:.1}%)",
            self.pipelines,
            self.translated,
            pct(self.translated, self.pipelines),
            self.seen,
            self.seen_translated,
            pct(self.seen_translated, self.seen)
        )
        .ok();
        writeln!(
            w,
            "first blocker of each pipeline that is not translated (pipelines by refusal key):"
        )
        .ok();
        let mut rows: Vec<_> = self.blockers.iter().collect();
        rows.sort_by_key(|(_, t)| std::cmp::Reverse(t.count));
        for (key, t) in rows.iter().take(top) {
            writeln!(w, "  {:>7}  {key}  (for example {})", t.count, t.example).ok();
        }
        if rows.len() > top {
            writeln!(w, "  ... and {} more keys", rows.len() - top).ok();
        }
        out
    }
}

/// Translates every pipeline whose programs are all in `programs`. A hash that is not in the
/// corpus is a refusal of its own (`container:missing-program`). Returns the census.
pub fn run(
    programs: &HashMap<u64, StageProgram>,
    lists: &PipelineLists,
    threads: usize,
) -> PipelineCensus {
    let pipelines: Vec<_> = lists.seen.iter().collect();
    let unknown = RootSignature::default();
    let defaults = Options::default();
    let parts = crate::corpus::parallel(
        pipelines.len(),
        threads,
        &PipelineCensus::default,
        |acc, i| {
            let (p, &seen) = pipelines[i];
            let key = translate_pipeline(p, programs, &lists.roots, &unknown, &defaults).err();
            acc.add(i, p, seen, key);
        },
    );
    let mut all = PipelineCensus::default();
    for p in parts {
        all.merge(p);
    }
    all
}

fn translate_pipeline(
    pipeline: &Pipeline,
    programs: &HashMap<u64, StageProgram>,
    roots: &HashMap<u64, ManifestRoot>,
    unknown: &RootSignature,
    defaults: &Options,
) -> Result<(), String> {
    let (root, options) = match &pipeline.manifest {
        Some(p) => {
            let root = p
                .root
                .and_then(|id| roots.get(&id))
                .ok_or("adapter:missing-root")?;
            (&root.signature, &root.options)
        }
        None => (unknown, defaults),
    };
    let stages = pipeline
        .stages
        .iter()
        .map(|(tag, hash)| {
            let p = programs.get(hash).ok_or("container:missing-program")?;
            let expected = match tag.as_str() {
                "VS" => Stage::Vertex,
                "HS" => Stage::Hull,
                "DS" => Stage::Domain,
                "GS" => Stage::Geometry,
                "PS" => Stage::Pixel,
                "CS" => Stage::Compute,
                "MS" => Stage::Mesh,
                "AS" => Stage::Amplification,
                _ => return Err("adapter:stage-mismatch"),
            };
            if p.stage != expected {
                return Err("adapter:stage-mismatch");
            }
            Ok(p.clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    crate::translate_probe(&stages, root, options, &CountingHost::default())
        .map(|_| ())
        .map_err(|r| r.key())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_manifest_root_parameters_and_layer_views() {
        let root = parse_root(concat!(
            "2 0xc01 table:0:t.2.1.1.0.8+b.0.1.4294967295.1.0 table:5: ",
            "const:5:3.0.4 uav:1:7.2 srv:0:0.3 cbv:0:1.15 srv:0:0.15 srv:0:1.15 ",
            "sampler:5:4.0"
        ))
        .unwrap();
        assert_eq!(root.signature.flags, 0xc01);
        assert_eq!(root.signature.params.len(), 5);
        assert_eq!(
            root.signature.params[0].kind,
            RootParamKind::Table(vec![
                RootRange {
                    class: RegisterClass::Srv,
                    base_register: 2,
                    space: 1,
                    count: 1,
                    offset: 0
                },
                RootRange {
                    class: RegisterClass::Cbv,
                    base_register: 0,
                    space: 1,
                    count: u32::MAX,
                    offset: 1
                },
            ])
        );
        assert_eq!(
            root.signature.params[2],
            RootParam {
                visibility: 5,
                kind: RootParamKind::Constants {
                    register: 3,
                    space: 0,
                    count: 4
                },
            }
        );
        assert_eq!(
            root.signature.static_samplers,
            [StaticSampler {
                register: 4,
                space: 0,
                visibility: 5,
            }]
        );
        assert_eq!(
            root.options.mirror,
            Some(MirrorAbi {
                ..MirrorAbi::default()
            })
        );
        assert!(parse_root("2 0x0 table:0:t.1.1.1.1").is_none());
        assert!(parse_root("2 0x0").unwrap().options.mirror.is_none());
    }

    #[test]
    fn reads_the_layer_line() {
        let l = "d3d12_xs: refused pipeline 0, hr=0x8007000E, graphics, slot 86: VS e5fcb71cbca948c4 miss, PS 7ba0e3e2345b876d miss, targets [28], depth 20";
        let p = parse_line(l).unwrap();
        assert_eq!(p.id(), "VS:e5fcb71cbca948c4+PS:7ba0e3e2345b876d");
        assert!(parse_line("PS is fine, VS 12 too short").is_none());
        assert!(parse_line("CS H miss").is_none());
        assert_eq!(
            parse_line("slot 5: CS 00000000000000ff miss")
                .unwrap()
                .stages
                .len(),
            1
        );
    }
}
