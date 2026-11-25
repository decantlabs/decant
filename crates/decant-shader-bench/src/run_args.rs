//! Command line of `decant_shader_run`, unchanged from the C++ bench:
//!
//! ```text
//! decant_shader_run <vs.spv> <ps.spv> <sortie.f32> [--buffer <binding> <fichier>]... [--push <indice> <valeur>]...
//!              [--texture <binding image> <binding sampler>] [--layout <fichier>] [--no-robust]
//!              [--texture-file <octets RGBA8> <largeur> <hauteur>] [--pipeline-cache <fichier>]
//! ```
//!
//! Files named by `--buffer` and `--texture-file` are read while the options are read, as the
//! C++ did, so a missing file is reported before a later unknown option.

use std::ffi::{OsStr, OsString};

use crate::Exit;
use crate::cfmt::{atoi, strtoul0};
use crate::layout::{self, LayoutEntry};

pub const USAGE: &str =
    "usage : decant_shader_run <vs.spv> <ps.spv> <sortie.f32> [--buffer <binding> <fichier>]...";

/// A storage buffer of set 0 from `--buffer`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bound {
    pub binding: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunArgs {
    pub vs: OsString,
    pub ps: OsString,
    pub output: OsString,
    pub bound: Vec<Bound>,
    /// The 32 push constant words.
    pub user_data: [u32; 32],
    /// `--texture`: image binding, negative when absent.
    pub texture_binding: i32,
    pub sampler_binding: i32,
    pub layout_path: Option<OsString>,
    /// `robustBufferAccess` and `robustImageAccess`, as for the published figures.
    pub robust: bool,
    /// `--texture-file`: a real RGBA8 image for 2D samplers, with its size.
    pub real_texture: Vec<u8>,
    pub real_w: u32,
    pub real_h: u32,
    /// `--pipeline-cache`: a `VkPipelineCache` file seeded before the pipeline is created and
    /// written back after, so the driver really compiles the pipeline.
    pub pipeline_cache_path: Option<OsString>,
}

/// Reads the options after the three paths. `read` loads a file (the bench uses
/// [`crate::read_file`]).
pub fn parse(
    args: &[OsString],
    mut read: impl FnMut(&OsStr) -> Result<Vec<u8>, Exit>,
) -> Result<RunArgs, Exit> {
    if args.len() < 4 {
        return Err(Exit::new(1, USAGE));
    }
    let text = |i: usize| args[i].to_string_lossy();
    let mut run = RunArgs {
        vs: args[1].clone(),
        ps: args[2].clone(),
        output: args[3].clone(),
        bound: Vec::new(),
        user_data: [0; 32],
        texture_binding: -1,
        sampler_binding: -1,
        layout_path: None,
        robust: true,
        real_texture: Vec::new(),
        real_w: 0,
        real_h: 0,
        pipeline_cache_path: None,
    };
    let argc = args.len();
    let mut i = 4;
    while i < argc {
        let option = text(i);
        if option == "--no-robust" {
            run.robust = false;
            i += 1;
            continue;
        }
        if option == "--texture-file" && i + 3 < argc {
            run.real_texture = read(&args[i + 1])?;
            run.real_w = atoi(&text(i + 2)) as u32;
            run.real_h = atoi(&text(i + 3)) as u32;
            i += 4;
            continue;
        }
        let two = i + 2 < argc;
        if option == "--buffer" && two {
            let binding = atoi(&text(i + 1)) as u32;
            run.bound.push(Bound {
                binding,
                data: read(&args[i + 2])?,
            });
            i += 3;
        } else if option == "--texture" && two {
            run.texture_binding = atoi(&text(i + 1));
            run.sampler_binding = atoi(&text(i + 2));
            i += 3;
        } else if option == "--push" && two && (0..32).contains(&atoi(&text(i + 1))) {
            // The C++ only checked `< 32`, a negative index wrote outside the array (undefined
            // behaviour) and is refused here as an unknown option.
            run.user_data[atoi(&text(i + 1)) as usize] = strtoul0(&text(i + 2));
            i += 3;
        } else if option == "--pipeline-cache" && i + 1 < argc {
            run.pipeline_cache_path = Some(args[i + 1].clone());
            i += 2;
        } else if option == "--layout" && i + 1 < argc {
            run.layout_path = Some(args[i + 1].clone());
            i += 2;
        } else {
            return Err(Exit::new(1, format!("option inconnue : {option}")));
        }
    }
    Ok(run)
}

/// Reads and checks the `--layout` file, then refuses a set 0 binding that `--buffer` or
/// `--texture` already takes, in the C++ order.
pub fn load_layout(
    run: &RunArgs,
    read_text: impl FnOnce(&OsStr) -> Option<Vec<u8>>,
) -> Result<Vec<LayoutEntry>, Exit> {
    let entries = match &run.layout_path {
        // An empty path meant no layout in the C++ (`layout_path.empty()`).
        Some(path) if !path.is_empty() => {
            let bytes = read_text(path).ok_or_else(|| {
                Exit::new(
                    2,
                    format!("lecture impossible : {}", path.to_string_lossy()),
                )
            })?;
            layout::parse(&bytes)
        }
        _ => Vec::new(),
    };
    layout::check(&entries, run.robust)?;
    for e in entries.iter().filter(|e| e.set == 0) {
        let texture = run.texture_binding >= 0
            && (run.texture_binding as u32 == e.binding || run.sampler_binding as u32 == e.binding);
        if run.bound.iter().any(|b| b.binding == e.binding) || texture {
            return Err(Exit::new(
                2,
                format!(
                    "binding {} du jeu 0 donne deux fois (--layout et --buffer/--texture)",
                    e.binding
                ),
            ));
        }
    }
    Ok(entries)
}
