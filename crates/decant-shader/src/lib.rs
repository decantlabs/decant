//! `decant-shader`: console shader programs (GFX10 machine code in FSSM containers) to HLSL
//! in one step, at the moment the title creates a pipeline, with the real root signature and
//! every stage of the pipeline in hand. DXC then makes DXIL, which runs on native Direct3D 12
//! and on vkd3d-proton.
//!
//! Written clean room: from Mesa's ACO sources (MIT), the LLVM AMDGPU backend, public AMD and
//! Microsoft documents and this project's own measurements. See the README of the crate.
//!
//! The passes, in the order [`translate`] runs them for each stage program:
//!
//! | Pass | Module | From | To |
//! |---|---|---|---|
//! | decode and graph | [`decode`], [`cfg`] | words | basic blocks of [`decode::Insn`] |
//! | lane masks | [`exec`] | graph | which scalar pairs are masks |
//! | lift | [`lift`] | instructions | [`ir::Stmt`] per block |
//! | structure | [`structure`] | blocks | [`ir::Node`] tree |
//! | entry state | [`stage`] | signature | prologue statements |
//! | resources | [`resource`] | raw descriptor uses | [`ir::ResId`] with an origin, then a placement |
//! | exports and inputs | [`stage`] | export records | output writes |
//! | print | [`hlsl`] | function | HLSL text |
//!
//! A program that a pass cannot carry is refused with a [`Refusal`] whose key is stable. The
//! census tool (`decant_shader_direct_census`) counts a corpus by those keys.

pub mod api;
pub mod asm;
pub mod cache;
pub mod census;
pub mod cfg;
pub mod container;
pub mod corpus;
pub mod decode;
pub mod exec;
pub mod hlsl;
pub mod ir;
pub mod lift;
pub mod refusal;
pub mod resource;
pub mod stage;
pub mod structure;
pub mod tables;

pub use api::*;
pub use refusal::{Refusal, RefusalKind};

/// Changes whenever the translation of any program can change. Part of the cache key.
pub const TRANSLATOR_VERSION: u32 = 6;

/// The program of the stage that feeds the pixel stage, when the pipeline has one.
fn before_pixel(stages: &[StageProgram]) -> Option<&StageProgram> {
    // The last stage before the rasteriser wins: geometry, then domain, mesh, vertex.
    [Stage::Geometry, Stage::Domain, Stage::Mesh, Stage::Vertex]
        .iter()
        .find_map(|s| stages.iter().find(|p| p.stage == *s))
}

/// Translates one stage program up to the IR, without placing resources or printing. The
/// census uses the pieces to see how far a program gets.
pub fn lift_program(program: &StageProgram) -> Result<ir::Function, Refusal> {
    let graph = cfg::build(&program.code)?;
    let masks = exec::analyse(&graph)?;
    let mut lifter = lift::Lifter::new(program.stage, &masks).with_relative(&graph);
    let mut blocks = Vec::with_capacity(graph.blocks.len());
    for (i, block) in graph.blocks.iter().enumerate() {
        if i == 0 {
            lifter.stmts = stage::prologue(program);
        }
        for insn in &block.insns {
            lift::lift(&mut lifter, insn)?;
        }
        blocks.push(lifter.take());
    }
    if blocks.iter().flatten().any(|stmt| {
        matches!(stmt, ir::Stmt::Set { value: ir::Expr::Call(h, _), .. }
            if std::ptr::eq(*h, &lift::wave::DPP_ADD)
                || std::ptr::eq(*h, &lift::wave::PERMLANE)
                || std::ptr::eq(*h, &lift::wave::BPERMUTE)
                || std::ptr::eq(*h, &lift::wave::DPP8_MOV))
    }) {
        lift::wave::check_dpp_context(&graph, &masks)?;
    }
    let mut body = structure::build(&graph, blocks)?;
    if matches!(program.stage, Stage::Compute | Stage::Mesh) {
        lift::wave::lower_swizzles(program.stage, &mut body)?;
    }
    Ok(ir::Function {
        body,
        resources: Vec::new(),
        groupshared_dwords: 0,
        lanes: masks.lanes(),
    })
}

fn translate_stage(
    program: &StageProgram,
    before: Option<&StageProgram>,
    root: &RootSignature,
    options: &Options,
    host: &dyn Host,
) -> Result<TranslatedStage, Refusal> {
    let mut func = lift_program(program)?;
    stage::prepare(program, &mut func)?;
    resource::trace(&mut func, program)?;
    // A read at an address in vector registers that the resource pass did not resolve.
    let mut unresolved = None;
    hlsl::expr::for_each_expr(&func, &mut |e| {
        if let ir::Expr::VectorLoad { pc, .. } = e {
            unresolved.get_or_insert(*pc);
        }
    });
    if let Some(pc) = unresolved {
        return Err(Refusal::new(RefusalKind::Resource("raw-address"), "a global load").at(pc));
    }
    let mut iface = stage::interface(program, before)?;
    // A stage that takes the group index (compute) numbers its lanes by it unless it says otherwise.
    if iface.lane_index.is_none() && iface.threads.is_some() {
        iface.lane_index = Some(format!("(decant_gi % {}u)", func.lanes.max(1)));
    }
    stage::finish(program, &iface, &mut func)?;
    resource::dce(&mut func);
    let needs = resource::place(&mut func, program, root, options, host)?;
    let mut header = vec![format!(
        "decant-shader {TRANSLATOR_VERSION}: {}",
        program.name
    )];
    if options.listing
        && let Ok(insns) =
            decode::decode_all(&program.code[container::SIGNATURE_WORDS.min(program.code.len())..])
    {
        header.extend(decode::listing(&insns).lines().map(str::to_owned));
    }
    let text = hlsl::print(
        &func,
        &hlsl::Context {
            iface: &iface,
            mirror: options.mirror,
            header: &header,
            code: &program.code,
        },
    )?;
    if options.shader_model_minor < 6 && text.lines().any(|line| line.starts_with("[WaveSize(")) {
        return Err(Refusal::new(
            RefusalKind::Stage("dpp-shader-model"),
            "native DPP requires Shader Model 6.6",
        ));
    }
    let text = stage::entry(program, &iface, text)?;
    // A buffer of the mirror ABI is used when its name occurs after its declaration.
    let used = |name: &str| text.matches(name).count() > 1;
    let mut needs = needs;
    needs.heap_map = used(hlsl::HEAP_MAP);
    needs.view_shadow = used(hlsl::VIEW_SHADOW);
    let resources = func
        .resources
        .iter()
        .filter_map(|r| {
            r.placement.map(|placement| BoundResource {
                name: r.name.clone(),
                placement,
            })
        })
        .collect();
    Ok(TranslatedStage {
        stage: program.stage,
        hlsl: text,
        entry: hlsl::ENTRY.to_owned(),
        profile: format!(
            "{}_6_{}",
            program.stage.profile_prefix(),
            options.shader_model_minor
        ),
        // IEEE strictness: the translation relies on exact float results and on not a number
        // behaving as on the console. Denormals are flushed to zero: no program of the title
        // sets the denormal bits of the mode register, and many programs of every stage use
        // the output modifier, which the hardware only honours with 32 bit denormals flushed
        // (measured on the corpus). The pipeline's own float mode is not in the
        // code, so this is an inference until the container field is found.
        dxc_args: vec!["-Gis".to_owned(), "-denorm".to_owned(), "ftz".to_owned()],
        resources,
        needs,
    })
}

/// The cheap answer to "is this pipeline worth translating": it looks at nothing but the
/// stage kinds and the first words of the code, so it costs microseconds. `Ok` does not promise
/// that [`translate`] succeeds, an `Err` promises that it would fail with that refusal.
/// [`translate`] runs it first, so a pipeline with a stage that is not written yet never
/// reaches the decoder. A layer calls it where it decides whether to take a pipeline at all.
pub fn can_translate(stages: &[StageProgram]) -> Result<(), Refusal> {
    for program in stages {
        stage::supported(program.stage)?;
        if !container::has_signature(&program.code) {
            return Err(
                Refusal::new(RefusalKind::Container("no-signature"), "").in_stage(program.stage)
            );
        }
    }
    Ok(())
}

/// Translates the stage programs of one pipeline to HLSL.
///
/// All stages of the pipeline are given together, in any order, because they are linked here:
/// the input struct of the pixel stage is made from the output signature of the stage before
/// it. `root` is the title's root signature for the pipeline, without the parameters the layer
/// appends. An empty `root` is allowed and means "unknown".
///
/// The first program that cannot be translated refuses the whole pipeline: a pipeline with one
/// console stage left cannot be created. [`can_translate`] is asked first.
pub fn translate(
    stages: &[StageProgram],
    root: &RootSignature,
    options: &Options,
    host: &dyn Host,
) -> Result<Translation, Refusal> {
    can_translate(stages)?;
    translate_probe(stages, root, options, host)
}

/// [`translate`] without the check of [`can_translate`], for the census: a program of a stage
/// that is not written yet is still decoded, lifted and traced, so that its first blocker is
/// the first real gap and not the stage.
pub fn translate_probe(
    stages: &[StageProgram],
    root: &RootSignature,
    options: &Options,
    host: &dyn Host,
) -> Result<Translation, Refusal> {
    let before = before_pixel(stages);
    let mut out = Vec::with_capacity(stages.len());
    for program in stages {
        // The program on the other side of a link: the stage in front, for the stages whose
        // input struct is its output signature, and the pixel program behind a mesh program,
        // which only keeps the output members that program takes.
        let feeds = match program.stage {
            Stage::Pixel => before,
            Stage::Geometry => [Stage::Domain, Stage::Vertex]
                .iter()
                .find_map(|s| stages.iter().find(|p| p.stage == *s)),
            Stage::Mesh => stages.iter().find(|p| p.stage == Stage::Pixel),
            _ => None,
        };
        out.push(
            translate_stage(program, feeds, root, options, host)
                .map_err(|r| r.in_stage(program.stage))?,
        );
    }
    Ok(Translation { stages: out })
}

/// [`translate`], then each stage through `compiler`.
pub fn translate_to_dxil(
    stages: &[StageProgram],
    root: &RootSignature,
    options: &Options,
    host: &dyn Host,
    compiler: &dyn Compiler,
) -> Result<Vec<CompiledStage>, Refusal> {
    let translation = translate(stages, root, options, host)?;
    translation
        .stages
        .into_iter()
        .map(|s| {
            let dxil = compiler
                .compile(&s.hlsl, &s.entry, &s.profile, &s.dxc_args)
                .map_err(|message| Refusal::new(RefusalKind::Compile, message).in_stage(s.stage))?;
            Ok(CompiledStage {
                stage: s.stage,
                dxil,
                resources: s.resources,
                needs: s.needs,
            })
        })
        .collect()
}
