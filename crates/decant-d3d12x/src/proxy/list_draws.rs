//! Draws, dispatches and indirect execution of the newer console list layout: `DrawIndexedX` (83),
//! `DispatchX` (80) and `ExecuteIndirect` (61).

use super::*;

type DrawIndexed = unsafe extern "system" fn(Object, u32, u32, u32, i32, u32);
type SetIndexBuffer = unsafe extern "system" fn(Object, *const [u8; 16]);

/// Whether the newer layout is in effect, else `slot` is logged as an unknown one.
pub(super) fn newer_or_unknown(this: *mut Proxy, slot: u32) -> bool {
    if crate::abi::newer_layout() {
        return true;
    }
    unknown_slot(this.cast(), Kind::List, slot);
    false
}

const DXGI_FORMAT_R32_UINT: u32 = 42;
const DXGI_FORMAT_R16_UINT: u32 = 57;

/// The bytes of one index in an index format, `None` for a format that is not an index format.
pub(super) fn index_stride(format: u32) -> Option<u32> {
    match format {
        DXGI_FORMAT_R16_UINT => Some(2),
        DXGI_FORMAT_R32_UINT => Some(4),
        _ => None,
    }
}

/// A `D3D12_INDEX_BUFFER_VIEW` of `count` indices of `format` at `va`, `None` for a format that is not
/// an index format or a size that does not fit.
pub(super) fn index_view_bytes(va: u64, format: u32, count: u32) -> Option<[u8; 16]> {
    let size = count.checked_mul(index_stride(format)?)?;
    let mut v = [0u8; 16];
    v[0..8].copy_from_slice(&va.to_le_bytes());
    v[8..12].copy_from_slice(&size.to_le_bytes());
    v[12..16].copy_from_slice(&format.to_le_bytes());
    Some(v)
}

/// Console list slot 83: `DrawIndexedX(index address, format, count, flags)` in the newer layout,
/// `CopyTextureRegion` with an 8th argument in the older one. The first five words are laid out alike
/// in both (Win64: four registers, then the stack), so one method serves both.
#[allow(clippy::too_many_arguments)]
pub(super) extern "system" fn list_slot83(this: *mut Proxy, a1: u64, a2: u32, a3: u32, a4: u32, a5: *const c_void, a6: *const c_void, a7: u64) {
    if !crate::abi::newer_layout() {
        return list_copy_texture_x(this, a1 as usize as *const c_void, a2, a3, a4, a5, a6, a7);
    }
    draw_indexed_x(this, a1, a2, a3, a4);
}

/// One instance of `count` indices read from `va`. The index buffer the title had bound stays bound.
fn draw_indexed_x(this: *mut Proxy, va: u64, format: u32, count: u32, flags: u32) {
    static N: AtomicI32 = AtomicI32::new(0);
    static REFUSED: AtomicI32 = AtomicI32::new(0);
    if count == 0 || crate::pso_diag::skip(this as usize) {
        return;
    }
    let bytes = index_stride(format).map_or(1, |stride| u64::from(stride) * u64::from(count));
    let Some(view) = index_view_bytes(super::pc_gpu_address(va, bytes).unwrap_or(va), format, count) else {
        if log_due(&REFUSED) {
            xs_log!("console list slot 83 (DrawIndexedX {va:#x}, format {format}, count {count}): not an index format, skipped");
        }
        return;
    };
    if log_due(&N) {
        xs_log!("console list slot 83 (DrawIndexedX, newer layout): {count} indices of format {format} at {va:#x}, flags {flags:#x}");
    }
    // SAFETY: `this` is a live list proxy; PC 43 is IASetIndexBuffer and 13 is DrawIndexedInstanced.
    unsafe {
        let pc = (*this).pc;
        super::list_mirror::prepare(this, crate::user_mirror::Bind::Graphics);
        let _ = crate::rt_diag::draw(this as usize, "DrawIndexedX", count);
        crate::flicker_probe::draw(this as usize, crate::flicker_probe::Draw::IndexedAt { va, format, count });
        crate::uav_barrier_merge::mark_pending(this as usize);
        method::<SetIndexBuffer>(pc, 43)(pc, &view);
        method::<DrawIndexed>(pc, 13)(pc, count, 1, 0, 0, 0);
        match crate::list_sync::index_view(this as usize) {
            Some(bound) => method::<SetIndexBuffer>(pc, 43)(pc, &bound),
            None => method::<SetIndexBuffer>(pc, 43)(pc, std::ptr::null()),
        }
    }
}

/// Console list slot 80 (newer layout): `DispatchX(x, y, z, flags)`, PC `Dispatch` (14).
pub(super) extern "system" fn list_dispatch_x(this: *mut Proxy, x: u32, y: u32, z: u32, flags: u32) {
    type Dispatch = unsafe extern "system" fn(Object, u32, u32, u32);
    static N: AtomicI32 = AtomicI32::new(0);
    if !newer_or_unknown(this, 80) || crate::pso_diag::skip(this as usize) {
        return;
    }
    if log_due(&N) {
        xs_log!("console list slot 80 (DispatchX, newer layout): ({x}, {y}, {z}), flags {flags:#x}");
    }
    // SAFETY: `this` is a live list proxy.
    unsafe {
        let pc = (*this).pc;
        super::list_mirror::prepare(this, crate::user_mirror::Bind::Compute);
        crate::rt_diag::count(&crate::rt_diag::DISPATCHES);
        super::list_mirror::describe_dispatch(this, crate::draw_diag::dispatch(this as usize, x, y, z));
        crate::uav_barrier_merge::mark_pending(this as usize);
        method::<Dispatch>(pc, 14)(pc, x, y, z);
    }
}

/// Console list slot 59: `ExecuteIndirect` in the older layout. In the newer layout the method moved
/// to 61 and what is at 59 is not known, so the call is logged and skipped (the arguments would not
/// be ExecuteIndirect's).
#[allow(clippy::too_many_arguments)]
pub(super) extern "system" fn list_slot59(this: *mut Proxy, a1: Object, a2: u32, a3: Object, a4: u64, a5: Object, a6: u64) {
    if !crate::abi::newer_layout() {
        return list_execute_indirect(this, a1, a2, a3, a4, a5, a6);
    }
    unknown_slot(this.cast(), Kind::List, 59);
}

/// Console list slot 61 (newer layout): `ExecuteIndirect(signature, max count, argument buffer,
/// offset, count buffer, count offset)`, the method the older layout has at 59. Measured in a title:
/// `(signature, 1, buffer, offset, NULL, 0)`.
#[allow(clippy::too_many_arguments)]
pub(super) extern "system" fn list_execute_indirect_x(this: *mut Proxy, sig: Object, max: u32, arg: Object, arg_off: u64, count: Object, count_off: u64) {
    if !newer_or_unknown(this, 61) {
        return;
    }
    list_execute_indirect(this, sig, max, arg, arg_off, count, count_off);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_index_view_holds_the_size_in_bytes_of_the_indices() {
        let v = index_view_bytes(0x1234_5678_9000, DXGI_FORMAT_R16_UINT, 30).unwrap();
        assert_eq!(u64::from_le_bytes(v[0..8].try_into().unwrap()), 0x1234_5678_9000);
        assert_eq!(u32::from_le_bytes(v[8..12].try_into().unwrap()), 60);
        assert_eq!(u32::from_le_bytes(v[12..16].try_into().unwrap()), 57);
        let w = index_view_bytes(0, DXGI_FORMAT_R32_UINT, 30).unwrap();
        assert_eq!(u32::from_le_bytes(w[8..12].try_into().unwrap()), 120);
    }

    #[test]
    fn only_16_and_32_bit_index_formats_make_a_view() {
        assert_eq!(index_view_bytes(0x1000, 28, 3), None);
        assert_eq!(index_view_bytes(0x1000, 0, 3), None);
        assert_eq!(index_view_bytes(0x1000, DXGI_FORMAT_R32_UINT, u32::MAX), None, "the size does not fit");
    }
}
