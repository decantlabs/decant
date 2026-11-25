//! The rasterizer state calls of a console list that the draw probe notes on their way to the PC list.

use super::*;

/// Console list slot 21: `RSSetViewports(count, viewports)` (PC 21), relayed and noted for the draw probe.
pub(super) extern "system" fn list_rs_set_viewports(this: *mut Proxy, count: u32, viewports: *const [f32; 6]) {
    type SetViewports = unsafe extern "system" fn(Object, u32, *const [f32; 6]);
    // SAFETY: `this` is a live list proxy, `viewports` is null or points to `count` viewports.
    unsafe {
        crate::draw_diag::viewports(this as usize, count, if count > 0 && !viewports.is_null() { Some(viewports.read_unaligned()) } else { None });
        let pc = (*this).pc;
        method::<SetViewports>(pc, 21)(pc, count, viewports);
    }
}

/// Console list slot 22: `RSSetScissorRects(count, rectangles)` (PC 22), relayed and noted for the draw probe.
pub(super) extern "system" fn list_rs_set_scissor_rects(this: *mut Proxy, count: u32, rects: *const [i32; 4]) {
    type SetScissors = unsafe extern "system" fn(Object, u32, *const [i32; 4]);
    // SAFETY: `this` is a live list proxy, `rects` is null or points to `count` rectangles.
    unsafe {
        crate::draw_diag::scissors(this as usize, count, if count > 0 && !rects.is_null() { Some(rects.read_unaligned()) } else { None });
        let pc = (*this).pc;
        method::<SetScissors>(pc, 22)(pc, count, rects);
    }
}
