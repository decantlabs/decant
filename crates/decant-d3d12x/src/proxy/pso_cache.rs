//! Console device slots 57 and 67, the load side of the pipeline cache (56 and 66 serialize):
//! `DeserializeGraphicsPipelineStateX` (57) and
//! `DeserializeComputePipelineStateX` (67), `(cache entry, riid, out)`. The title ships a pipeline
//! cache in the console driver's serialized form and tries every entry of it while it boots (measured:
//! tens of thousands of graphics and compute entries).
//!
//! What the title does with the answer (measured on the title executable):
//! * 57 is called from a loop over the cache entries (exe+0x2477578). On success the pipeline goes
//!   into the title's pipeline map. On any failure the loop goes on with the next entry. Nothing
//!   else happens on a failure: no retry, no logging, no rebuild of the entry.
//! * 67 is called from a lookup that returns the pipeline or null (exe+0x24800d5). On null the
//!   caller creates the pipeline itself through device slot 62 (exe+0x247fe66), with zero
//!   extensions, which `lazy_pso` answers at once.
//! * Both pass the result to one HRESULT check (exe+0x4e9b660) that singles out only the device
//!   lost codes (`DXGI_ERROR_DEVICE_REMOVED`, `_HUNG`, `_RESET`, `_DRIVER_INTERNAL_ERROR`). Every
//!   other failure is the same miss to the title, so no answer makes it faster than another.
//!
//! The answer is `D3D12_ERROR_DRIVER_VERSION_MISMATCH`, the public D3D12 code for serialized
//! pipeline data that another driver made, which is what these entries are. It is the answer, not
//! a gap, so it is logged the first times and not counted in `DECANT_CALLS`. A cache of our own
//! plugs in at [`cached_pipeline`].

use super::*;

/// `D3D12_ERROR_DRIVER_VERSION_MISMATCH`: the serialized data comes from another driver.
pub(super) const DRIVER_VERSION_MISMATCH: HRESULT = 0x887E_0002_u32 as HRESULT;

/// Which cache method the title called.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheKind {
    /// Slot 57.
    Graphics,
    /// Slot 67.
    Compute,
}

impl CacheKind {
    fn slot(self) -> u32 {
        match self {
            CacheKind::Graphics => 57,
            CacheKind::Compute => 67,
        }
    }

    fn name(self) -> &'static str {
        match self {
            CacheKind::Graphics => "DeserializeGraphicsPipelineStateX",
            CacheKind::Compute => "DeserializeComputePipelineStateX",
        }
    }
}

/// The hook of a pipeline cache of our own: the pipeline object for the title's cache
/// entry `entry`, or `None` for a miss. A cache would key the entry by the content of the
/// serialized data it points to and answer the pipeline (a `lazy_pso` object made from the
/// description saved by an earlier run). There is none yet, so every entry is a miss.
fn cached_pipeline(_kind: CacheKind, _entry: *const c_void) -> Option<Object> {
    None
}

/// One deserialize call: a hit from [`cached_pipeline`] with `S_OK`, else `out` cleared and
/// [`DRIVER_VERSION_MISMATCH`].
///
/// # Safety
/// `out` is null or writable.
unsafe fn deserialize(kind: CacheKind, entry: *const c_void, out: *mut *mut c_void) -> HRESULT {
    static LOGGED: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
    if out.is_null() {
        return E_POINTER;
    }
    if let Some(pipeline) = cached_pipeline(kind, entry) {
        // SAFETY: by the contract above.
        unsafe { *out = pipeline };
        return S_OK;
    }
    // A miss is the answer, not a gap: it is not counted in DECANT_CALLS.
    if log_due(&LOGGED[kind as usize]) {
        xs_log!("console device slot {} ({}, entry {}): no pipeline cache of ours yet, a miss (0x{:08X}); the title creates the pipeline from its description", kind.slot(), kind.name(), p(entry), DRIVER_VERSION_MISMATCH as u32);
    }
    // SAFETY: by the contract above.
    unsafe { *out = std::ptr::null_mut() };
    DRIVER_VERSION_MISMATCH
}

/// Console device slot 57: `DeserializeGraphicsPipelineStateX(cache entry, riid, out)`.
pub(super) extern "system" fn dev_deserialize_graphics(_this: *mut Proxy, entry: *const c_void, _riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: `out` is the caller's result pointer.
    unsafe { deserialize(CacheKind::Graphics, entry, out) }
}

/// Console device slot 67: `DeserializeComputePipelineStateX(cache entry, riid, out)`.
pub(super) extern "system" fn dev_deserialize_compute(_this: *mut Proxy, entry: *const c_void, _riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: `out` is the caller's result pointer.
    unsafe { deserialize(CacheKind::Compute, entry, out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The title's HRESULT check (exe+0x4e9b660): `hr - DXGI_ERROR_DEVICE_REMOVED` with bits 0, 1, 2
    /// and 27 of 0x8000007 set is device lost.
    fn title_sees_device_lost(hr: HRESULT) -> bool {
        let at = (hr as u32).wrapping_add(0x7785_FFFB);
        hr < 0 && at <= 0x1B && (0x0800_0007u32 >> at) & 1 != 0
    }

    #[test]
    fn the_miss_is_a_failure_the_title_does_not_take_for_device_loss() {
        assert!(!title_sees_device_lost(DRIVER_VERSION_MISMATCH));
        for lost in [0x887A_0005u32, 0x887A_0006, 0x887A_0007, 0x887A_0020] {
            assert!(title_sees_device_lost(lost as HRESULT), "{lost:#x}");
        }
    }

    #[test]
    fn every_entry_is_a_miss_and_clears_the_result() {
        let mut marker = 0u8;
        let mut out: *mut c_void = (&raw mut marker).cast();
        // SAFETY: `out` is writable.
        assert_eq!(unsafe { deserialize(CacheKind::Graphics, std::ptr::null(), &mut out) }, DRIVER_VERSION_MISMATCH);
        assert!(out.is_null());
        let mut out: *mut c_void = (&raw mut marker).cast();
        // SAFETY: as above.
        assert_eq!(unsafe { deserialize(CacheKind::Compute, std::ptr::null(), &mut out) }, DRIVER_VERSION_MISMATCH);
        assert!(out.is_null());
        // SAFETY: a null result pointer is refused before anything is written.
        assert_eq!(unsafe { deserialize(CacheKind::Compute, std::ptr::null(), std::ptr::null_mut()) }, E_POINTER);
    }
}
