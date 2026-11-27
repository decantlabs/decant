//! Predication, top of pipe writes, KickoffX, dynamic depth bias, clear-batch hints and the
//! ignored state calls of the newer console list layout (slots 75, 87, 90 to 92, 95, 96, 101, 126,
//! 131, 132 and 142).

use super::list_draws::newer_or_unknown;
use super::*;

type SetPredicationAt = unsafe extern "system" fn(Object, Object, u64, u32);
type DepthBias = unsafe extern "system" fn(Object, f32, f32, f32);
type GetListType = unsafe extern "system" fn(Object) -> u32;

const D3D12_COMMAND_LIST_TYPE_DIRECT: u32 = 0;
const D3D12_COMMAND_LIST_TYPE_COMPUTE: u32 = 2;
const D3D12_RESOURCE_BARRIER_TYPE_ALIASING: u32 = 1;

/// True for PC command list types that accept UAV and aliasing barriers (`DIRECT` and `COMPUTE`).
/// `BUNDLE` forbids `ResourceBarrier` altogether, and `COPY` forbids UAV and aliasing barriers.
pub(super) fn supports_uav_and_aliasing_barriers(list_type: u32) -> bool {
    matches!(list_type, D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE)
}

/// `[D3D12_RESOURCE_BARRIER_TYPE_UAV(NULL), D3D12_RESOURCE_BARRIER_TYPE_ALIASING(NULL, NULL)]`
/// in the 32-byte public `D3D12_RESOURCE_BARRIER` layout.
pub(super) fn global_uav_and_aliasing_barriers() -> [[u8; BARRIER_BYTES]; 2] {
    let uav = crate::list_sync::global_uav_barrier();
    let mut aliasing = [0u8; BARRIER_BYTES];
    aliasing[0..4].copy_from_slice(&D3D12_RESOURCE_BARRIER_TYPE_ALIASING.to_le_bytes());
    [uav, aliasing]
}

/// Records `[UAV(NULL), ALIASING(NULL, NULL)]` on PC command list `pc` when its type is
/// `DIRECT` or `COMPUTE`. Returns whether the barriers were recorded.
///
/// # Safety
/// `pc` is a live PC `ID3D12GraphicsCommandList`.
pub(super) unsafe fn record_global_uav_and_aliasing_barriers(pc: Object) -> bool {
    // SAFETY: by the contract above; PC slot 8 is `ID3D12CommandList::GetType` and slot 26 is `ResourceBarrier`.
    unsafe {
        if !supports_uav_and_aliasing_barriers(method::<GetListType>(pc, 8)(pc)) {
            return false;
        }
        let barriers = global_uav_and_aliasing_barriers();
        method::<Barriers>(pc, 26)(pc, barriers.len() as u32, barriers.as_ptr().cast());
        true
    }
}

/// Applies a resolved predication target to PC command list `pc`. On `DIRECT` and `COMPUTE` lists,
/// enabling predication records `[UAV(NULL), ALIASING(NULL, NULL)]` first so prior UAV writes and
/// `title_mem` reserved-buffer aliases are visible to subsequent predicated operations. `BUNDLE`
/// and `COPY` lists do not support `SetPredication` in PC D3D12 (calling it poisons `Close` with
/// `E_FAIL`), so they are skipped.
///
/// # Safety
/// `pc` is a live PC `ID3D12GraphicsCommandList`, and any resource in `target` is a live PC buffer.
unsafe fn apply_predication(pc: Object, target: Option<(Object, u64, u32)>) {
    // SAFETY: by the contract above; PC slot 8 is `GetType`, 26 is `ResourceBarrier`, and 55 is `SetPredication`.
    unsafe {
        if !supports_uav_and_aliasing_barriers(method::<GetListType>(pc, 8)(pc)) {
            return;
        }
        match target {
            Some((res, at, op)) => {
                let barriers = global_uav_and_aliasing_barriers();
                method::<Barriers>(pc, 26)(pc, barriers.len() as u32, barriers.as_ptr().cast());
                method::<SetPredicationAt>(pc, 55)(pc, res, at, op);
            }
            None => method::<SetPredicationAt>(pc, 55)(pc, std::ptr::null_mut(), 0, 0),
        }
    }
}

/// The IID of `ID3D12GraphicsCommandList9`, the interface that has `RSSetDepthBias`.
pub(super) const IID_GRAPHICS_COMMAND_LIST9: GUID = GUID::from_u128(0x34ed2808_ffe6_4c2b_b11a_cabd2b0c59e1);
/// The PC slot of `ID3D12GraphicsCommandList9::RSSetDepthBias` (81 is
/// `ID3D12GraphicsCommandList8::OMSetFrontAndBackStencilRef`; `pc_slots` checks it).
pub(super) const PC_RS_SET_DEPTH_BIAS: usize = 82;
/// Flag bits of `Write64BitValueTopOfPipeX` that ask for a GPU timestamp instead of a value.
const TOP_TIMESTAMPS: u32 = 0xC;

/// The PC predication operation of a console one (0 equal zero, 1 not equal zero).
pub(super) fn predication_op(raw: u32) -> Option<u32> {
    (raw <= 1).then_some(raw)
}

/// A float the PC accepts: a value that is not finite becomes 0.
pub(super) fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Console list slot 75 (newer layout, older 73): a per list switch taken from the bound program.
/// A retail title calls it right after `SetPipelineState` and `SetGraphicsRootSignature`, only when
/// the value changes, with one argument (bit 6 of a flags byte of the program, so 0 or 1). It
/// selects the pixel shader depth order. On PC the driver takes that order from the shader itself,
/// so there is nothing to set and the value is only logged the first times.
pub(super) extern "system" fn list_depth_order_x(this: *mut Proxy, enable: u32) {
    static N: AtomicI32 = AtomicI32::new(0);
    if newer_or_unknown(this, 75) && log_due(&N) {
        xs_log!("console list slot 75 (pixel shader depth order, newer layout): {}, nothing to set on PC", enable & 1);
    }
}

/// Console list slot 87 (newer layout): `SetPredicationBufferX(desc)` with `desc` null or
/// `{ u64 address; u32 operation }`. The 8 bytes at the address are the predicate. Before enabling
/// predication on a Direct or Compute list, global `[UAV(NULL), ALIASING(NULL, NULL)]` barriers
/// flush prior UAV writes and `title_mem` reserved-buffer aliases; predication itself is modal across
/// subsequent commands, and buffers in `COMMON` (both `CreateReservedResource` and default-heap
/// buffers) implicitly promote to `PREDICATION` while upload buffers stay in `GENERIC_READ`. An
/// address the proxy does not know switches predication off, so that draws are not lost.
pub(super) extern "system" fn list_set_predication_buffer_x(this: *mut Proxy, desc: *const u8) {
    static N: AtomicI32 = AtomicI32::new(0);
    if !newer_or_unknown(this, 87) {
        return;
    }
    // SAFETY: `this` is a live list proxy; `desc` is null or 12 readable bytes by the method's contract.
    unsafe {
        let pc = (*this).pc;
        if desc.is_null() {
            apply_predication(pc, None);
            return;
        }
        let va = desc.cast::<u64>().read_unaligned();
        let op = desc.add(8).cast::<u32>().read_unaligned();
        let target = if va.is_multiple_of(8) { predication_op(op).and_then(|op| list_mem::resolve(va, 8).map(|(r, o)| (r, o, op))) } else { None };
        match target {
            Some((res, at, op)) => {
                if log_due(&N) {
                    xs_log!("console list slot 87 (SetPredicationBufferX, newer layout): predicate at {va:#x}, operation {op}");
                }
                apply_predication(pc, Some((res, at, op)));
            }
            None => {
                if log_due(&N) {
                    xs_log!("console list slot 87 (SetPredicationBufferX): {va:#x} operation {op} is not a predicate in a known buffer, predication switched off");
                }
                apply_predication(pc, None);
            }
        }
    }
}

/// Console list slot 90 (newer layout): `Write32BitValueTopOfPipeX(address, value, flags)`. Done by the
/// CPU after the GPU, as the bottom of pipe writes of slots 71 and 72.
pub(super) extern "system" fn list_write32_top(this: *mut Proxy, va: u64, value: u32, _flags: u32) {
    if newer_or_unknown(this, 90) {
        bottom_write(this, va, &[value], 90);
    }
}

/// Console list slot 91 (newer layout): `Write64BitValueTopOfPipeX(address, value, flags)`. The
/// timestamp form (flags `TOP_TIMESTAMPS`) gets the GPU clock after the list ran, as slot 72 does.
pub(super) extern "system" fn list_write64_top(this: *mut Proxy, va: u64, value: u64, flags: u32) {
    if !newer_or_unknown(this, 91) {
        return;
    }
    if flags & TOP_TIMESTAMPS != 0 {
        return bottom_write(this, va, &[], 91);
    }
    bottom_write(this, va, &[value as u32, (value >> 32) as u32], 91);
}

/// Console list slot 92 (newer layout): `Write32BitValueBottomOfShaderX(address, value, stage)`, done
/// by the CPU after the GPU.
pub(super) extern "system" fn list_write32_bottom_of_shader(this: *mut Proxy, va: u64, value: u32, _stage: u32) {
    if newer_or_unknown(this, 92) {
        bottom_write(this, va, &[value], 92);
    }
}

/// Console list slot 95 (newer layout): `KickoffX()`, the title asks for the commands so far to be
/// sent to the GPU. They run when the list is executed, so this is only logged.
pub(super) extern "system" fn list_kickoff_x(this: *mut Proxy) {
    static N: AtomicI32 = AtomicI32::new(0);
    if newer_or_unknown(this, 95) && log_due(&N) {
        xs_log!("console list slot 95 (KickoffX, newer layout): nothing to do, the commands run when the list is executed");
    }
}

/// Console list slot 96 (newer layout): `RSSetDepthBiasX(bias, clamp, slope)` as three floats, PC
/// `ID3D12GraphicsCommandList9::RSSetDepthBias` when the list has that interface, else logged once.
pub(super) extern "system" fn list_set_depth_bias_x(this: *mut Proxy, bias: f32, clamp: f32, slope: f32) {
    static N: AtomicI32 = AtomicI32::new(0);
    static MISSING: AtomicBool = AtomicBool::new(false);
    if !newer_or_unknown(this, 96) {
        return;
    }
    let (bias, clamp, slope) = (finite_or_zero(bias), finite_or_zero(clamp), finite_or_zero(slope));
    if log_due(&N) {
        xs_log!("console list slot 96 (RSSetDepthBiasX, newer layout): bias {bias}, clamp {clamp}, slope {slope}");
    }
    // SAFETY: `this` is a live list proxy; the interface found by QueryInterface is released again.
    unsafe {
        let pc = (*this).pc;
        let mut l9: Object = std::ptr::null_mut();
        if query_object(pc, &IID_GRAPHICS_COMMAND_LIST9, &mut l9) < 0 || l9.is_null() {
            if !MISSING.swap(true, Ordering::SeqCst) {
                xs_log!("console list slot 96 (RSSetDepthBiasX): the PC list has no ID3D12GraphicsCommandList9, the depth bias is not applied");
            }
            return;
        }
        method::<DepthBias>(l9, PC_RS_SET_DEPTH_BIAS)(l9, bias, clamp, slope);
        method::<Release>(l9, 2)(l9);
    }
}

/// Console list slot 101 (newer layout, older 99): a two-argument toggle `(mode: u32, enable: u32)`
/// bracketing three or more consecutive `ClearRenderTargetView` / `ClearDepthStencilView` calls as
/// a console-only clear-overlap optimization hint. PC D3D12 has no command-list toggle around clears
/// and runs every clear unconditionally, so this is a silent no-op on the newer layout.
pub(super) extern "system" fn list_clear_hint_x(this: *mut Proxy, _mode: u32, _enable: u32) {
    let _ = newer_or_unknown(this, 101);
}

/// Console list slot 126 (newer layout, older 124): `RSSetShadingRate` in the console form. The
/// PC list gets no variable rate, the proxy reads no argument, and the call is only logged the
/// first times.
pub(super) extern "system" fn list_shading_rate_x(this: *mut Proxy) {
    static N: AtomicI32 = AtomicI32::new(0);
    if !newer_or_unknown(this, 126) {
        return;
    }
    if log_due(&N) {
        xs_log!("console list slot 126 (shading rate, newer layout): ignored, no variable rate on PC");
    }
}

/// Console list slots 131 (top of pipe) and 132 (bottom of pipe), newer layout:
/// `PipelineSync(flags, control, address, size)`. Console titles use pipeline syncs to flush and
/// serialize GPU caches between passes in place of PC `ResourceBarrier` calls, so on Direct and
/// Compute lists this records global `[UAV(NULL), ALIASING(NULL, NULL)]` barriers (`BUNDLE` and
/// `COPY` lists skip them). Measured calls name address 0; a sync that names memory still records
/// the barriers while logging that its memory write is not implemented.
fn pipeline_sync(this: *mut Proxy, slot: u32, flags: u32, control: u32, address: u64, size: u64) {
    static TOP_LOGS: AtomicI32 = AtomicI32::new(0);
    static BOTTOM_LOGS: AtomicI32 = AtomicI32::new(0);
    if !newer_or_unknown(this, slot) {
        return;
    }
    // SAFETY: `this` is a live command list proxy whose PC list is live.
    let recorded = unsafe { record_global_uav_and_aliasing_barriers((*this).pc) };
    let logs = if slot == 131 { &TOP_LOGS } else { &BOTTOM_LOGS };
    if !log_due(logs) {
        return;
    }
    let action = if recorded { "global UAV and aliasing barriers recorded on PC list" } else { "skipped on non-direct/compute list" };
    if address == 0 {
        xs_log!("console list slot {slot} (pipeline sync, flags {flags:#x}, control {control:#x}, size {size:#x}): no memory named, {action}");
    } else {
        xs_log!("console list slot {slot} (pipeline sync, flags {flags:#x}, control {control:#x}): names memory at {address:#x}, {action} (memory write not performed)");
    }
}

/// Console list slot 131: the top of pipe sync, see [`pipeline_sync`].
pub(super) extern "system" fn list_sync_top(this: *mut Proxy, flags: u32, control: u32, address: u64, size: u64) {
    pipeline_sync(this, 131, flags, control, address, size);
}

/// Console list slot 132: the bottom of pipe sync, see [`pipeline_sync`].
pub(super) extern "system" fn list_sync_bottom(this: *mut Proxy, flags: u32, control: u32, address: u64, size: u64) {
    pipeline_sync(this, 132, flags, control, address, size);
}

/// Console list slot 142 (newer layout): a state call with one `u64` argument whose result the
/// title does not use. Its purpose is not measured, so the argument is not read and the call is
/// only logged the first times.
pub(super) extern "system" fn list_state_142(this: *mut Proxy) {
    static N: AtomicI32 = AtomicI32::new(0);
    if !newer_or_unknown(this, 142) {
        return;
    }
    if log_due(&N) {
        xs_log!("console list slot 142 (state call, newer layout): one u64 argument not read, ignored");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_two_predication_operations_are_taken() {
        assert_eq!(predication_op(0), Some(0));
        assert_eq!(predication_op(1), Some(1));
        assert_eq!(predication_op(2), None);
    }

    #[test]
    fn a_depth_bias_that_is_not_finite_becomes_zero() {
        assert_eq!(finite_or_zero(1.5), 1.5);
        assert_eq!(finite_or_zero(f32::NAN), 0.0);
        assert_eq!(finite_or_zero(f32::INFINITY), 0.0);
        assert_eq!(finite_or_zero(-0.25), -0.25);
    }

    #[test]
    fn uav_and_aliasing_barriers_match_d3d12_layout() {
        let [uav, aliasing] = global_uav_and_aliasing_barriers();
        assert_eq!(u32::from_le_bytes(uav[0..4].try_into().unwrap()), 2);
        assert!(uav[4..].iter().all(|&b| b == 0));
        assert_eq!(u32::from_le_bytes(aliasing[0..4].try_into().unwrap()), 1);
        assert!(aliasing[4..].iter().all(|&b| b == 0));

        #[cfg(windows)]
        {
            use windows::Win32::Graphics::Direct3D12::{D3D12_RESOURCE_BARRIER_TYPE_ALIASING, D3D12_RESOURCE_BARRIER_TYPE_UAV};
            assert_eq!(u32::from_le_bytes(uav[0..4].try_into().unwrap()), D3D12_RESOURCE_BARRIER_TYPE_UAV.0 as u32);
            assert_eq!(u32::from_le_bytes(aliasing[0..4].try_into().unwrap()), D3D12_RESOURCE_BARRIER_TYPE_ALIASING.0 as u32);
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum ListEvent {
        Barriers(Vec<[u8; BARRIER_BYTES]>),
        Predication(usize, u64, u32),
    }

    #[repr(C)]
    struct FakeList {
        vtbl: *const usize,
        list_type: u32,
        events: Vec<ListEvent>,
    }

    unsafe extern "system" fn fake_get_type(this: Object) -> u32 {
        // SAFETY: only `FakeList` objects carry this vtable.
        unsafe { (*this.cast::<FakeList>()).list_type }
    }

    unsafe extern "system" fn fake_barriers(this: Object, n: u32, barriers: *const c_void) {
        // SAFETY: only `FakeList` objects carry this vtable; `barriers` holds `n` barrier structs.
        unsafe {
            let list = &mut *this.cast::<FakeList>();
            let mut recorded = Vec::with_capacity(n as usize);
            for i in 0..n as usize {
                recorded.push(barriers.cast::<[u8; BARRIER_BYTES]>().add(i).read_unaligned());
            }
            list.events.push(ListEvent::Barriers(recorded));
        }
    }

    unsafe extern "system" fn fake_set_predication(this: Object, res: Object, off: u64, op: u32) {
        // SAFETY: only `FakeList` objects carry this vtable.
        unsafe {
            (*this.cast::<FakeList>()).events.push(ListEvent::Predication(res as usize, off, op));
        }
    }

    fn fake_list_vtable() -> [usize; 56] {
        let mut vtbl = [0usize; 56];
        vtbl[8] = fake_get_type as *const () as usize;
        vtbl[26] = fake_barriers as *const () as usize;
        vtbl[55] = fake_set_predication as *const () as usize;
        vtbl
    }

    #[test]
    fn pipeline_sync_records_uav_and_aliasing_barriers_on_direct_and_compute_only() {
        let vtbl = fake_list_vtable();
        let expected = global_uav_and_aliasing_barriers().to_vec();
        for (list_type, should_record) in [(0u32, true), (1, false), (2, true), (3, false)] {
            let mut fake = FakeList { vtbl: vtbl.as_ptr(), list_type, events: Vec::new() };
            let pc = std::ptr::from_mut(&mut fake).cast::<c_void>();
            // SAFETY: `fake` implements vtable slots 8 and 26.
            let recorded = unsafe { record_global_uav_and_aliasing_barriers(pc) };
            assert_eq!(recorded, should_record, "list_type {list_type}");
            if should_record {
                assert_eq!(fake.events, vec![ListEvent::Barriers(expected.clone())], "list_type {list_type}");
            } else {
                assert!(fake.events.is_empty(), "list_type {list_type}");
            }
        }
    }

    #[test]
    fn predication_emits_barriers_before_enable_and_skips_on_clear_or_copy() {
        let vtbl = fake_list_vtable();
        let expected = global_uav_and_aliasing_barriers().to_vec();
        let res = 0x1234_5000usize as Object;
        for (list_type, supported) in [(0u32, true), (1, false), (2, true), (3, false)] {
            let mut fake = FakeList { vtbl: vtbl.as_ptr(), list_type, events: Vec::new() };
            let pc = std::ptr::from_mut(&mut fake).cast::<c_void>();
            // SAFETY: `fake` implements vtable slots 8, 26 and 55.
            unsafe {
                apply_predication(pc, Some((res, 0x40, 1)));
                apply_predication(pc, None);
            }
            let want = if supported {
                vec![
                    ListEvent::Barriers(expected.clone()),
                    ListEvent::Predication(res as usize, 0x40, 1),
                    ListEvent::Predication(0, 0, 0),
                ]
            } else {
                Vec::new()
            };
            assert_eq!(fake.events, want, "list_type {list_type}");
        }
    }

    #[test]
    fn slot_101_and_146_are_dispatched_and_clear_hint_is_a_silent_noop() {
        assert_eq!(custom(Kind::List, 101), Some(list_clear_hint_x as *const () as usize));
        assert_eq!(custom(Kind::List, 146), Some(enhanced_barriers::list_barrier_x as *const () as usize));
        let vtbl = fake_list_vtable();
        let mut fake = FakeList { vtbl: vtbl.as_ptr(), list_type: 0, events: Vec::new() };
        let mut proxy = Proxy {
            vtbl: std::ptr::null(),
            pc: std::ptr::from_mut(&mut fake).cast(),
            refs: AtomicI32::new(1),
            kind: Kind::List as i32,
        };
        // Both `(2, 1)` enable and `(2, 0)` disable leave the PC command list untouched.
        list_clear_hint_x(&mut proxy, 2, 1);
        list_clear_hint_x(&mut proxy, 2, 0);
        assert!(fake.events.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn pipeline_sync_and_predication_close_cleanly_on_d3d12_lists() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::core::Interface;

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 65536,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        for (name, device) in crate::view_gpu_tests::devices() {
            let mut reserved: Option<ID3D12Resource> = None;
            // SAFETY: valid device and buffer description.
            unsafe { device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut reserved) }
                .unwrap_or_else(|e| panic!("{name}: CreateReservedResource failed: {e}"));
            let reserved = reserved.expect("reserved buffer");

            for (list_type, expect_barriers) in [
                (D3D12_COMMAND_LIST_TYPE_DIRECT, true),
                (D3D12_COMMAND_LIST_TYPE_COMPUTE, true),
                (D3D12_COMMAND_LIST_TYPE_COPY, false),
            ] {
                // SAFETY: standard D3D12 allocator and command list creation on a live device.
                let (alloc, list): (ID3D12CommandAllocator, ID3D12GraphicsCommandList) = unsafe {
                    let alloc = device.CreateCommandAllocator(list_type).expect("allocator");
                    let list = device.CreateCommandList(0, list_type, &alloc, None).expect("command list");
                    (alloc, list)
                };
                // SAFETY: `list` and `reserved` are live D3D12 objects for this device.
                unsafe {
                    assert_eq!(record_global_uav_and_aliasing_barriers(list.as_raw()), expect_barriers, "{name}: {list_type:?}");
                    apply_predication(list.as_raw(), Some((reserved.as_raw(), 0, 0)));
                    apply_predication(list.as_raw(), None);
                    list.Close().unwrap_or_else(|e| panic!("{name}: Close failed for {list_type:?}: {e}"));
                }
                drop((list, alloc));
            }
        }
    }
}
