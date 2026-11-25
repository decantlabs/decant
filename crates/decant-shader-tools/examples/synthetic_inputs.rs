//! Writes the invented inputs of the tests into a folder, for running the Python scripts and the
//! Rust tools side by side on them: a measurement folder of SPIR-V modules, the reflection sample
//! and bench images, correct and deliberately wrong. Nothing in them comes from a game file.
//!
//! Usage: `cargo run -p decant-shader-tools --example synthetic_inputs -- <folder>`

#[allow(dead_code)]
#[path = "../src/pyfmt.rs"]
mod pyfmt;
#[allow(dead_code)]
#[path = "../src/synthetic.rs"]
mod synthetic;

// synthetic.rs reaches the half float helpers through crate::pyfmt.
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("usage: synthetic_inputs <folder>"));
    synthetic::write_measurement(&out.join("measure")).expect("write the measurement");
    std::fs::write(out.join("reflect.spv"), synthetic::Module::reflect_sample().bytes()).unwrap();
    let images = [
        ("blit_good.f32", synthetic::blit_image([1.0; 4], false)),
        ("blit_scaled.f32", synthetic::blit_image([0.5, 2.0, 1.0, 0.25], false)),
        ("blit_wrong.f32", synthetic::blit_image([1.0, 1.0, 0.5, 1.0], false)),
        ("blit_undrawn.f32", synthetic::blit_image([1.0; 4], true)),
        ("texture_good.f32", synthetic::texture_image(false, 0.0)),
        ("texture_linked.f32", synthetic::texture_image(true, 0.0)),
        ("texture_wrong.f32", synthetic::texture_image(false, 0.01)),
    ];
    for (name, data) in images {
        std::fs::write(out.join(name), data).unwrap();
    }
}
