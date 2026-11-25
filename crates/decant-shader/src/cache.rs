//! The cache key of a translation.
//!
//! The disk cache itself belongs to the runtime (`d3d12-xs`). This file only says what a
//! translation depends on, so that the key is defined next to the translator and changes with
//! it: the translator version, the options, the root signature, and of every stage the code,
//! the signatures, the declared resources and the binding table. Names are left out.
//!
//! Sampler indices come from the host and end up in the text. The runtime's sampler registry
//! must therefore give the same index for the same words in every run, or the runtime adds its
//! own registry generation to the key through `host_tag`.

use sha1::{Digest, Sha1};

use crate::api::{
    Options, RegisterClass, RootParamKind, RootSignature, SigElement, StageProgram, WaveMode,
};

struct Feed(Sha1);

impl Feed {
    fn u32(&mut self, v: u32) {
        self.0.update(v.to_le_bytes());
    }

    fn class(&mut self, c: RegisterClass) {
        self.u32(c as u32);
    }

    fn sig(&mut self, sig: &[SigElement]) {
        self.u32(sig.len() as u32);
        for e in sig {
            self.u32(e.semantic.len() as u32);
            self.0.update(e.semantic.as_bytes());
            for v in [
                e.index,
                e.system_value,
                e.component_type,
                e.register,
                u32::from(e.mask),
                u32::from(e.rw_mask),
            ] {
                self.u32(v);
            }
        }
    }
}

/// The key of translating `stages` with `root` and `options`: 20 bytes.
pub fn key(
    stages: &[StageProgram],
    root: &RootSignature,
    options: &Options,
    host_tag: u64,
) -> [u8; 20] {
    let mut f = Feed(Sha1::new());
    f.u32(crate::TRANSLATOR_VERSION);
    f.0.update(host_tag.to_le_bytes());
    match options.mirror {
        Some(m) => [
            1,
            m.register,
            m.space,
            m.heap_map.unwrap_or(u32::MAX),
            m.view_shadow.unwrap_or(u32::MAX),
        ]
        .into_iter()
        .for_each(|v| f.u32(v)),
        None => f.u32(0),
    }
    f.u32(match options.wave {
        WaveMode::Scalar => 0,
        WaveMode::Native { lanes } => 0x100 | u32::from(lanes),
        WaveMode::Emulated => 2,
    });
    f.u32(u32::from(options.shader_model_minor));
    f.u32(u32::from(options.listing));
    f.u32(root.flags);
    f.u32(root.params.len() as u32);
    for p in &root.params {
        f.u32(p.visibility);
        match &p.kind {
            RootParamKind::Table(ranges) => {
                f.u32(0);
                f.u32(ranges.len() as u32);
                for r in ranges {
                    f.class(r.class);
                    [r.count, r.base_register, r.space, r.offset]
                        .into_iter()
                        .for_each(|v| f.u32(v));
                }
            }
            RootParamKind::Constants {
                register,
                space,
                count,
            } => [1, *register, *space, *count]
                .into_iter()
                .for_each(|v| f.u32(v)),
            RootParamKind::Descriptor {
                class,
                register,
                space,
            } => {
                f.u32(2);
                f.class(*class);
                f.u32(*register);
                f.u32(*space);
            }
        }
    }
    f.u32(root.static_samplers.len() as u32);
    for s in &root.static_samplers {
        [s.register, s.space, s.visibility]
            .into_iter()
            .for_each(|v| f.u32(v));
    }
    f.u32(stages.len() as u32);
    for p in stages {
        f.u32(p.stage as u32);
        f.u32(p.code.len() as u32);
        p.code.iter().for_each(|w| f.u32(*w));
        f.sig(&p.inputs);
        f.sig(&p.outputs);
        f.u32(p.resources.len() as u32);
        for r in &p.resources {
            f.class(r.class);
            [r.lower, r.upper, r.space, r.psv_type, r.psv_kind]
                .into_iter()
                .for_each(|v| f.u32(v));
        }
        f.u32(p.binding_table.len() as u32);
        for b in &p.binding_table {
            f.class(b.class);
            [b.register, u32::from(b.sgpr), b.word_offset]
                .into_iter()
                .for_each(|v| f.u32(v));
        }
        p.threads.iter().for_each(|v| f.u32(*v));
        f.u32(p.lds_bytes);
        match p.geometry {
            Some(g) => [
                1,
                g.input_primitive,
                g.output_topology,
                g.stream_mask,
                g.max_vertices,
            ]
            .into_iter()
            .for_each(|v| f.u32(v)),
            None => f.u32(0),
        }
        f.u32(p.user_data.len() as u32);
        for u in &p.user_data {
            let words = match u.kind {
                crate::api::UserDataKind::Table => [0, 0, 0],
                crate::api::UserDataKind::Constants { register, count } => [1, register, count],
                crate::api::UserDataKind::RootSrv { register } => [2, register, 0],
                crate::api::UserDataKind::ResourceHeap => [3, 0, 0],
                crate::api::UserDataKind::SamplerHeap => [4, 0, 0],
            };
            f.u32(u32::from(u.sgpr));
            words.into_iter().for_each(|v| f.u32(v));
        }
        f.u32(p.param_exports.len() as u32);
        for e in &p.param_exports {
            [
                u32::from(e.slot),
                u32::from(e.kind),
                e.register,
                e.sixteen_bit_register,
                u32::from(e.flags),
            ]
            .into_iter()
            .for_each(|v| f.u32(v));
        }
        f.u32(p.pixel_inputs.len() as u32);
        for i in &p.pixel_inputs {
            let high = i.high_register.unwrap_or(u32::MAX);
            [
                u32::from(i.attr),
                u32::from(i.kind),
                i.register,
                high,
                u32::from(i.channels),
                u32::from(i.flags),
            ]
            .into_iter()
            .for_each(|v| f.u32(v));
        }
    }
    f.0.finalize().into()
}
