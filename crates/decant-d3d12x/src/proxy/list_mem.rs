//! Memory copies and fills of the newer console list layout: `CopyMemoryToMemoryX` (62) and
//! `FillMemoryWith32/64/128BitValueX` (63 to 65).
//!
//! The title names memory by address. Here an address leads to a PC buffer when it lies in a buffer
//! the title placed (device slots 44 and 133, by the address it chose or the GPU address the PC gave
//! it), in upload memory of list slot 148, or in title memory `xmem.dll` maps or commits (the window
//! of `title_mem`). Anything else, for example a buffer created by `CreateCommittedResource`, is not
//! known to the proxy and the call is logged once and skipped. What a copy or fill writes into the
//! window reaches the GPU's copy of title memory only: the CPU does not see it before the write back
//! of title memory v2 phase 2.
//!
//! The copies are `CopyBufferRegion` calls without barriers: the title's own barriers for these
//! buffers are the console's, and the PC state of a buffer is not tracked here.

use std::sync::Mutex;

use super::list_draws::newer_or_unknown;
use super::*;

type CopyRegion = unsafe extern "system" fn(Object, Object, u64, Object, u64, u64);

/// The bytes of one fill chunk, which is also the size of a pattern buffer.
const PATTERN_BYTES: usize = 256 << 10;
/// The most bytes one fill call writes.
const MAX_FILL: u64 = 256 << 20;
/// The most pattern buffers kept (each is 256 KiB of upload memory).
const MAX_PATTERNS: usize = 32;

/// `(offset, length)` of the copies that write `size` bytes in pieces of at most `chunk`.
pub(super) fn fill_chunks(size: u64, chunk: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut at = 0u64;
    while at < size {
        let len = chunk.min(size - at);
        out.push((at, len));
        at += len;
    }
    out
}

/// The 16 bytes a fill repeats for the value bytes `value` (4, 8 or 16 of them).
pub(super) fn pattern(value: &[u8]) -> [u8; 16] {
    let mut p = [0u8; 16];
    for (i, b) in p.iter_mut().enumerate() {
        *b = value[i % value.len()];
    }
    p
}

/// The PC buffer and the offset in it for `[va, va + len)`.
pub(super) fn resolve(va: u64, len: u64) -> Option<(Object, u64)> {
    crate::upload_mem::locate(va, len)
        .or_else(|| crate::views_at::lookup_span(va, len))
        .or_else(|| crate::title_mem::translate(va, len).map(|s| (s.resource, s.offset)))
        .map(|(res, off)| (res as Object, off))
}

fn skipped(what: &str, why: std::fmt::Arguments) {
    static N: AtomicI32 = AtomicI32::new(0);
    if log_due(&N) {
        xs_log!("console list {what}: skipped, {why}");
    }
}

/// Console list slot 62 (newer layout): `CopyMemoryToMemoryX(destination, source, size, flags)`.
pub(super) extern "system" fn list_copy_memory_x(this: *mut Proxy, dst: u64, src: u64, size: u64, flags: u32) {
    static N: AtomicI32 = AtomicI32::new(0);
    if !newer_or_unknown(this, 62) {
        return;
    }
    if size == 0 || !size.is_multiple_of(4) {
        return skipped("slot 62 (CopyMemoryToMemoryX)", format_args!("size {size:#x} is not a non-zero multiple of 4"));
    }
    let (Some((d, doff)), Some((s, soff))) = (resolve(dst, size), resolve(src, size)) else {
        return skipped("slot 62 (CopyMemoryToMemoryX)", format_args!("{src:#x} to {dst:#x}, {size:#x} bytes: an address is not in a buffer the proxy knows"));
    };
    if d == s && doff < soff + size && soff < doff + size {
        return skipped("slot 62 (CopyMemoryToMemoryX)", format_args!("{src:#x} to {dst:#x}, {size:#x} bytes: the ranges overlap"));
    }
    if log_due(&N) {
        xs_log!("console list slot 62 (CopyMemoryToMemoryX, newer layout): {src:#x} to {dst:#x}, {size:#x} bytes, flags {flags:#x}");
    }
    // SAFETY: `this` is a live list proxy; `d` and `s` are live PC buffers (PC 15 is CopyBufferRegion).
    unsafe {
        let pc = (*this).pc;
        method::<CopyRegion>(pc, 15)(pc, d, doff, s, soff, size);
    }
}

/// A buffer of `PATTERN_BYTES` bytes that repeats `pat`, made once per pattern.
///
/// # Safety
/// `list` is a live PC command list.
unsafe fn pattern_buffer(list: Object, pat: [u8; 16]) -> Option<Object> {
    static CACHE: Mutex<Vec<([u8; 16], usize)>> = Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, res)) = cache.iter().find(|(p, _)| *p == pat) {
        return Some(res as Object);
    }
    if cache.len() >= MAX_PATTERNS {
        return None;
    }
    // SAFETY: by the contract above; the mapping has PATTERN_BYTES writable bytes.
    unsafe {
        let (res, cpu, _) = list_gaps::new_upload_buffer(list, PATTERN_BYTES as u64)?;
        for chunk in std::slice::from_raw_parts_mut(cpu as *mut u8, PATTERN_BYTES).as_chunks_mut::<16>().0 {
            *chunk = pat;
        }
        crate::mem_owners::add_charge(crate::mem_owners::Charge {
            kind: crate::mem_owners::Kind::UploadChunk,
            segment: crate::mem_owners::Segment::Unknown,
            bytes: PATTERN_BYTES as u64,
        });
        cache.push((pat, res as usize));
        Some(res)
    }
}

fn fill(this: *mut Proxy, slot: u32, va: u64, size: u64, pat: [u8; 16], unit: u64) {
    static N: AtomicI32 = AtomicI32::new(0);
    let what = "FillMemoryWithValueX";
    if size == 0 || !size.is_multiple_of(unit) || size > MAX_FILL {
        return skipped(what, format_args!("slot {slot}: size {size:#x} is not a non-zero multiple of {unit} up to {MAX_FILL:#x}"));
    }
    if !va.is_multiple_of(unit) {
        return skipped(what, format_args!("slot {slot}: {va:#x} is not {unit} byte aligned"));
    }
    let Some((dst, off)) = resolve(va, size) else {
        return skipped(what, format_args!("slot {slot}: {va:#x}, {size:#x} bytes: not in a buffer the proxy knows"));
    };
    // SAFETY: `this` is a live list proxy; `dst` is a live PC buffer.
    unsafe {
        let pc = (*this).pc;
        let Some(src) = pattern_buffer(pc, pat) else {
            return skipped(what, format_args!("slot {slot}: no pattern buffer for {pat:x?}"));
        };
        if log_due(&N) {
            xs_log!("console list slot {slot} ({what}, newer layout): {size:#x} bytes at {va:#x} with {pat:x?}");
        }
        for (at, len) in fill_chunks(size, PATTERN_BYTES as u64) {
            method::<CopyRegion>(pc, 15)(pc, dst, off + at, src, 0, len);
        }
    }
}

/// Console list slot 63 (newer layout): `FillMemoryWith32BitValueX(address, size, value)`.
pub(super) extern "system" fn list_fill32_x(this: *mut Proxy, va: u64, size: u64, value: u32) {
    if newer_or_unknown(this, 63) {
        fill(this, 63, va, size, pattern(&value.to_le_bytes()), 4);
    }
}

/// Console list slot 64 (newer layout): `FillMemoryWith64BitValueX(address, size, value)`.
pub(super) extern "system" fn list_fill64_x(this: *mut Proxy, va: u64, size: u64, value: u64) {
    if newer_or_unknown(this, 64) {
        fill(this, 64, va, size, pattern(&value.to_le_bytes()), 8);
    }
}

/// Console list slot 65 (newer layout): `FillMemoryWith128BitValueX(address, size, value*)`, the value
/// is 16 bytes in memory.
pub(super) extern "system" fn list_fill128_x(this: *mut Proxy, va: u64, size: u64, value: *const [u8; 16]) {
    if !newer_or_unknown(this, 65) {
        return;
    }
    if value.is_null() {
        return skipped("slot 65 (FillMemoryWith128BitValueX)", format_args!("null value"));
    }
    // SAFETY: `value` is non-null and holds 16 bytes by the method's contract.
    let pat = unsafe { value.read_unaligned() };
    fill(this, 65, va, size, pat, 16);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fill_is_cut_into_pieces_that_add_up() {
        assert_eq!(fill_chunks(0, 16), vec![]);
        assert_eq!(fill_chunks(16, 16), vec![(0, 16)]);
        assert_eq!(fill_chunks(40, 16), vec![(0, 16), (16, 16), (32, 8)]);
        let total: u64 = fill_chunks(1_000_003, 4096).iter().map(|c| c.1).sum();
        assert_eq!(total, 1_000_003);
    }

    #[test]
    fn a_value_is_repeated_to_fill_sixteen_bytes() {
        assert_eq!(pattern(&0xAABB_CCDDu32.to_le_bytes()), [0xDD, 0xCC, 0xBB, 0xAA, 0xDD, 0xCC, 0xBB, 0xAA, 0xDD, 0xCC, 0xBB, 0xAA, 0xDD, 0xCC, 0xBB, 0xAA]);
        let q = 0x0102_0304_0506_0708u64.to_le_bytes();
        assert_eq!(&pattern(&q)[8..], &q);
        let x: [u8; 16] = std::array::from_fn(|i| i as u8);
        assert_eq!(pattern(&x), x);
    }
}
