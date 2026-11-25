//! Predication, top of pipe writes, KickoffX, dynamic depth bias and the ignored state calls of the
//! newer console list layout (slots 75, 87, 90 to 92, 95, 96, 126, 131, 132 and 142).

use super::list_draws::newer_or_unknown;
use super::*;

type SetPredicationAt = unsafe extern "system" fn(Object, Object, u64, u32);
type DepthBias = unsafe extern "system" fn(Object, f32, f32, f32);

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
/// `{ u64 address; u32 operation }`. The 8 bytes at the address are the predicate. The PC predicates
/// on the buffer where it is, so no barrier is recorded (the PC wants the buffer in the PREDICATION
/// state, which the proxy does not track). An address the proxy does not know switches predication
/// off, so that the draws are not lost.
pub(super) extern "system" fn list_set_predication_buffer_x(this: *mut Proxy, desc: *const u8) {
    static N: AtomicI32 = AtomicI32::new(0);
    if !newer_or_unknown(this, 87) {
        return;
    }
    // SAFETY: `this` is a live list proxy; `desc` is null or 12 readable bytes by the method's contract.
    unsafe {
        let pc = (*this).pc;
        let set = |res: Object, at: u64, op: u32| method::<SetPredicationAt>(pc, 55)(pc, res, at, op);
        if desc.is_null() {
            set(std::ptr::null_mut(), 0, 0);
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
                set(res, at, op);
            }
            None => {
                if log_due(&N) {
                    xs_log!("console list slot 87 (SetPredicationBufferX): {va:#x} operation {op} is not a predicate in a known buffer, predication switched off");
                }
                set(std::ptr::null_mut(), 0, 0);
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
/// `PipelineSync(flags, control, address, size)`. The PC list runs its commands in order and every
/// submission is waited for, so a sync without a memory address has nothing to do (like the texture
/// cache flush, slot 67). Measured calls name address 0. A sync that does name memory is logged and
/// not performed, because the write it asks for is not implemented.
fn pipeline_sync(this: *mut Proxy, slot: u32, flags: u32, control: u32, address: u64, size: u64) {
    static TOP_LOGS: AtomicI32 = AtomicI32::new(0);
    static BOTTOM_LOGS: AtomicI32 = AtomicI32::new(0);
    if !newer_or_unknown(this, slot) {
        return;
    }
    let logs = if slot == 131 { &TOP_LOGS } else { &BOTTOM_LOGS };
    if !log_due(logs) {
        return;
    }
    if address == 0 {
        xs_log!("console list slot {slot} (pipeline sync, flags {flags:#x}, control {control:#x}, size {size:#x}): no memory named, nothing to do on PC");
    } else {
        xs_log!("console list slot {slot} (pipeline sync, flags {flags:#x}, control {control:#x}): names memory at {address:#x}, not performed");
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
}
