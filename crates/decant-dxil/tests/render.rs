//! Graphics shaders drawn on a real D3D12 device (default adapter, WARP as fallback): a full
//! screen triangle with the vertex shader of test 10 and the pixel shaders of tests 11 and 12.
#![cfg(windows)]

mod common;

use common::gfx::{Attribute, TARGET_SIZE, Texture, pixel};
use common::gpu::open_device;
use common::{shaders, write_and_validate};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT};

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn near(got: [u8; 4], want: [u8; 4]) -> bool {
    got.iter().zip(want).all(|(g, w)| g.abs_diff(w) <= 1)
}

/// POSITION only: a triangle that covers the whole target.
fn position_triangle() -> Vec<u8> {
    floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0])
}

/// POSITION and TEXCOORD0 where the uv is the screen position, 0 to 1 from the top left.
fn uv_triangle() -> Vec<u8> {
    floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, 2.0, 1.0, -1.0, 3.0, 0.0, 1.0, 0.0, -1.0])
}

const POSITION: Attribute = Attribute { name: b"POSITION\0", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 };
const TEXCOORD: Attribute = Attribute { name: b"TEXCOORD\0", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 16 };

#[test]
fn a_constant_colour_triangle_is_drawn() {
    let Some(gpu) = open_device() else {
        println!("SKIPPED: no D3D12 device with Shader Model 6.6");
        return;
    };
    println!("device: {}", gpu.name);
    let vs = write_and_validate("vertex_passthrough", &shaders::vertex_passthrough());
    let ps = write_and_validate("pixel_constant", &shaders::pixel_constant());
    let out = gpu.render(&vs, &ps, &[POSITION], 16, &position_triangle(), 3, None).expect("render");
    for (x, y) in [(0, 0), (32, 32), (63, 63), (5, 58)] {
        assert!(near(pixel(&out, x, y), [255, 128, 64, 255]), "pixel {x},{y} is {:?}", pixel(&out, x, y));
    }
}

#[test]
fn a_texture_is_sampled_at_the_interpolated_coordinates() {
    let Some(gpu) = open_device() else {
        println!("SKIPPED: no D3D12 device with Shader Model 6.6");
        return;
    };
    println!("device: {}", gpu.name);
    let vs = write_and_validate("vertex_with_uv", &shaders::vertex_with_uv());
    let ps = write_and_validate("pixel_texture", &shaders::pixel_texture());
    // 2 by 2 texels: red, green on the top row, blue, white below.
    let texture = Texture { width: 2, height: 2, rgba: vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255] };
    let out = gpu.render(&vs, &ps, &[POSITION, TEXCOORD], 24, &uv_triangle(), 3, Some(&texture)).expect("render");
    let q = TARGET_SIZE / 4;
    assert_eq!(pixel(&out, q, q), [255, 0, 0, 255], "top left");
    assert_eq!(pixel(&out, 3 * q, q), [0, 255, 0, 255], "top right");
    assert_eq!(pixel(&out, q, 3 * q), [0, 0, 255, 255], "bottom left");
    assert_eq!(pixel(&out, 3 * q, 3 * q), [255, 255, 255, 255], "bottom right");
}

#[test]
fn discard_removes_the_right_half() {
    let Some(gpu) = open_device() else {
        println!("SKIPPED: no D3D12 device with Shader Model 6.6");
        return;
    };
    println!("device: {}", gpu.name);
    let vs = write_and_validate("vertex_with_uv", &shaders::vertex_with_uv());
    let ps = write_and_validate("pixel_discard", &shaders::pixel_discard());
    let out = gpu.render(&vs, &ps, &[POSITION, TEXCOORD], 24, &uv_triangle(), 3, None).expect("render");
    assert_eq!(pixel(&out, 8, 32), [0, 255, 0, 255], "kept");
    assert_eq!(pixel(&out, 56, 32), [0, 0, 0, 0], "discarded");
}
