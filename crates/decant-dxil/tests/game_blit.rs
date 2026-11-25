//! The title's colour blit (`PS_BlitColour` and its vertex shader) as the override cache builds
//! them, drawn on a D3D12 device with a known constant buffer and vertex colour.
//!
//! The containers belong to the game, so the test reads them from the folder that
//! `DECANT_SHADER_BLIT_OVERRIDE` names (the output of `decant_shader_dxil_cache`) and is skipped without it.
#![cfg(windows)]

mod common;

use std::path::PathBuf;

use common::gfx::{Attribute, Constants, TARGET_SIZE, pixel};
use common::gpu::open_device;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT};

/// FNV-1a 64 of the console containers, the names the override cache gives them.
const VERTEX: &str = "d2a238740e50f037";
const PIXEL: &str = "e2bcc91e569d6d13";

#[test]
fn the_colour_blit_draws_the_vertex_colour_times_the_constant() {
    let Some(dir) = std::env::var_os("DECANT_SHADER_BLIT_OVERRIDE").map(PathBuf::from) else {
        println!("SKIPPED: set DECANT_SHADER_BLIT_OVERRIDE to an override folder");
        return;
    };
    let vs = std::fs::read(dir.join(format!("{VERTEX}.dxil"))).expect("vertex container");
    let ps = std::fs::read(dir.join(format!("{PIXEL}.dxil"))).expect("pixel container");
    let Some(gpu) = open_device() else {
        println!("SKIPPED: no D3D12 device with Shader Model 6.6");
        return;
    };
    // The title's vertex layout: a float3 position, an RGBA8 colour and a float2 coordinate.
    let attributes = [
        Attribute { name: b"POSITION\0", index: 0, format: DXGI_FORMAT_R32G32B32_FLOAT, offset: 0 },
        Attribute { name: b"COLOR\0", index: 0, format: DXGI_FORMAT_R8G8B8A8_UNORM, offset: 12 },
        Attribute { name: b"TEXCOORD\0", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 16 },
    ];
    let mut vertices = Vec::new();
    for (x, y) in [(-1.0f32, -1.0f32), (3.0, -1.0), (-1.0, 3.0)] {
        for v in [x, y, 0.5] {
            vertices.extend_from_slice(&v.to_le_bytes());
        }
        vertices.extend_from_slice(&[255, 255, 255, 255]);
        vertices.extend_from_slice(&[0u8; 8]);
    }
    // The shader multiplies the colour by the four floats at dword 8 of b27 space 1.
    let mut cb = vec![0u8; 64];
    for (k, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
        cb[32 + 4 * k..36 + 4 * k].copy_from_slice(&v.to_le_bytes());
    }
    let constants = Constants { register: 27, space: 1, bytes: &cb };
    let out = gpu.render_with(&vs, &ps, &attributes, 24, &vertices, 3, None, Some(&constants)).expect("render");
    let centre = pixel(&out, TARGET_SIZE / 2, TARGET_SIZE / 2);
    println!("centre pixel {centre:?}");
    // 0.5 is 127.5 in eight bits, which the driver may round either way.
    assert!(matches!(centre, [255, 127 | 128, 64, 255]), "centre pixel {centre:?}");
}
