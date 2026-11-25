//! Every test shader through the validator of dxil.dll (skipped with a message when it is absent).

mod common;

use common::shaders;
use common::write_and_validate;

#[test]
fn empty_compute_validates() {
    write_and_validate("empty", &shaders::empty());
}

#[test]
fn thread_id_store_validates() {
    write_and_validate("thread_id_store", &shaders::thread_id_store());
}

#[test]
fn buffer_copy_validates() {
    write_and_validate("buffer_copy", &shaders::buffer_copy());
}

#[test]
fn arithmetic_validates() {
    write_and_validate("arithmetic", &shaders::arithmetic());
}

#[test]
fn loop_with_branch_validates() {
    write_and_validate("loop_with_branch", &shaders::loop_with_branch());
}

#[test]
fn groupshared_barrier_validates() {
    write_and_validate("groupshared_barrier", &shaders::groupshared_barrier());
}

#[test]
fn structured_copy_validates() {
    write_and_validate("structured_copy", &shaders::structured_copy());
}

#[test]
fn every_resource_class_and_kind_validates() {
    write_and_validate("resource_zoo", &shaders::resource_zoo());
}

#[test]
fn wave_ops_validate() {
    for size in [4, 32, 128] {
        write_and_validate(&format!("wave_ops_{size}"), &shaders::wave_ops(size));
    }
    let report = decant_dxil::write(&shaders::wave_ops(32)).unwrap().report;
    assert_eq!(report.wave_size, Some(32));
    assert!(!report.resource_heap);
    let mut bad = shaders::wave_ops(32);
    bad.stage = decant_dxil::ir::Stage::Compute { threads: [64, 1, 1], wave_size: Some(48) };
    assert!(decant_dxil::write(&bad).is_err(), "48 is not a power of two");
}

#[test]
fn heap_uav_validates_and_asks_for_the_heap_flag() {
    write_and_validate("heap_uav", &shaders::heap_uav());
    let report = decant_dxil::write(&shaders::heap_uav()).unwrap().report;
    assert!(report.resource_heap && !report.sampler_heap);
    assert_eq!(report.bound.len(), 1, "only the constant buffer is bound, the heap UAV is not");
}

#[test]
fn graphics_shaders_validate() {
    write_and_validate("vertex_passthrough", &shaders::vertex_passthrough());
    write_and_validate("vertex_with_uv", &shaders::vertex_with_uv());
    write_and_validate("pixel_constant", &shaders::pixel_constant());
    write_and_validate("pixel_texture", &shaders::pixel_texture());
    write_and_validate("pixel_discard", &shaders::pixel_discard());
    write_and_validate("pixel_zoo", &shaders::pixel_zoo());
    write_and_validate("pixel_heap_texture", &shaders::pixel_heap_texture());
    let r = decant_dxil::write(&shaders::pixel_heap_texture()).unwrap().report;
    assert!(r.resource_heap && r.sampler_heap && r.bound.is_empty());
    write_and_validate("vertex_zoo", &shaders::vertex_zoo());
    write_and_validate("vertex_heap_uav", &shaders::vertex_heap_uav());
}

#[test]
fn every_texture_operation_validates() {
    write_and_validate("texture_zoo", &shaders::texture_zoo());
    write_and_validate("texture_compute", &shaders::texture_compute());
}

#[test]
fn atomics_validate() {
    write_and_validate("atomics", &shaders::atomics());
}
